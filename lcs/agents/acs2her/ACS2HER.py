import logging
import random
from lcs import Perception
from lcs.agents.Agent import TrialMetrics
from lcs.agents.acs2er.ReplayMemory import ReplayMemory
from lcs.agents.acs2er.ReplayMemorySample import ReplayMemorySample
from lcs.strategies.action_selection.BestAction import BestAction
from lcs.agents.acs2 import ClassifiersList
from lcs.agents.acs2her import Configuration
from lcs.agents.Agent import Agent

logger = logging.getLogger(__name__)


class ACS2HER(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None) -> None:
        self.cfg = cfg
        self.population = population or ClassifiersList()
        self.replay_memory = ReplayMemory(max_size=cfg.er_buffer_size)

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, time,
                           current_trial=None) -> TrialMetrics:

        logger.debug("** Running trial explore ** ")

        # Initial conditions
        steps = 0
        state = env.reset()
        action = env.action_space.sample()
        last_reward = 0
        prev_state = Perception.empty()
        done = False

        current_trial_steps = []

        # if self.cfg.her_strategy in ['final', 'future', 'episode']:
        #     self.steps_buffer.clear()

        while not done:
            state = Perception(state)
            assert len(state) == self.cfg.classifier_length

            match_set = self.population.form_match_set(state)
            action = self.cfg.action_selector(match_set)
            logger.debug("\tExecuting action: [%d]", action)

            prev_state = Perception(state)
            raw_state, last_reward, done, _ = env.step(action)
            state = Perception(raw_state)

            current_trial_steps.append(
                [prev_state, action, last_reward, state, done])

            # Save experience in replay memory
            self.replay_memory.update(ReplayMemorySample(
                prev_state, action, last_reward, state, done))

            if len(self.replay_memory) >= self.cfg.er_min_samples:
                samples = random.sample(self.replay_memory,
                                        self.cfg.er_samples_number)
                for sample in samples:
                    er_match_set = self.population.form_match_set(
                        sample.state)
                    er_action_set = er_match_set.form_action_set(
                        sample.action)
                    er_next_match_set = self.population.form_match_set(
                        sample.next_state)
                    # Apply learning in the replied action set
                    ClassifiersList.apply_alp(
                        self.population,
                        er_next_match_set,
                        er_action_set,
                        sample.state,
                        sample.action,
                        sample.next_state,
                        time + steps,
                        self.cfg.theta_exp,
                        self.cfg)
                    ClassifiersList.apply_reinforcement_learning(
                        er_action_set,
                        sample.reward,
                        0 if sample.done
                        else er_next_match_set.get_maximum_fitness(),
                        self.cfg.beta,
                        self.cfg.gamma
                    )
                    if self.cfg.do_ga:
                        ClassifiersList.apply_ga(
                            time + steps,
                            self.population,
                            ClassifiersList() if sample.done
                            else er_next_match_set,
                            er_action_set,
                            sample.next_state,
                            self.cfg.theta_ga,
                            self.cfg.mu,
                            self.cfg.chi,
                            self.cfg.theta_as,
                            self.cfg.do_subsumption,
                            self.cfg.theta_exp)

            steps += 1

        for index, step in enumerate(current_trial_steps):
            state, action, reward, next_state, done = step

            new_goals = self.sample_goals(current_trial_steps, index)

            for new_goal in new_goals:
                new_reward = self.reward_function(state, new_goal)
                new_done = next_state == new_goal

                self.replay_memory.update(
                    ReplayMemorySample(state, action, new_reward,
                                       next_state, new_done))

        return TrialMetrics(steps, last_reward)

    def _run_trial_exploit(self, env, time=None,
                           current_trial=None) -> TrialMetrics:

        logger.debug("** Running trial exploit **")
        # Initial conditions
        steps = 0
        state = Perception(env.reset())

        last_reward = 0
        action_set = ClassifiersList()
        done = False

        while not done:
            match_set = self.population.form_match_set(state)

            if steps > 0:
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma)

            # Here when exploiting always choose best action
            action = BestAction(
                all_actions=self.cfg.number_of_possible_actions)(match_set)
            action_set = match_set.form_action_set(action)

            state, last_reward, done, _ = env.step(action)
            state = Perception(state)

            if done:
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.cfg.beta, self.cfg.gamma)

            steps += 1

        return TrialMetrics(steps, last_reward)

    def sample_goals(self, steps, index):
        k = self.cfg.her_new_goals_number
        # if self.cfg.her_strategy in ['episode', 'random']:
        #     steps = steps
        if self.cfg.her_strategy == 'future':
            steps = steps[index:]
        elif self.cfg.her_strategy == 'final':
            steps = [steps[-1]]

        if k <= len(steps):
            return [s[0] for s in random.sample(steps, k=k)]
        return [s[0] for s in steps]

    def reward_function(self, state, new_goal):
        return 0 if state == new_goal else self.cfg.her_penalty_reward
