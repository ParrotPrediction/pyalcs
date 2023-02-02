import logging
import random

from lcs.agents.Agent import Agent
from lcs.agents.Agent import TrialMetrics
from lcs.agents.acs2 import ClassifiersList
from lcs.agents.acs2er import ReplayMemorySample
from lcs.agents.acs2her import Configuration
from lcs.agents.acs2her.PerceptionExtensionBuilder import *
from lcs.agents.acs2her.ReplayBuffer import ReplayBuffer
from lcs.strategies.action_selection.BestAction import BestAction

logger = logging.getLogger(__name__)


class ACS2HERv2(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None) -> None:
        self.cfg = cfg
        self.population = population or ClassifiersList()
        self.memory = ReplayBuffer(max_size=cfg.er_buffer_size,
                                   samples_number=cfg.er_samples_number)
        self.additional_goals = []
        self.main_goal = Perception.empty()

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, time,
                           current_trial=None) -> TrialMetrics:

        logger.debug("** Running trial explore ** ")

        # Initial conditions
        state = env.reset()
        # action = env.action_space.sample()
        last_reward = 0
        # prev_state = Perception.empty()
        done = False

        trial_steps = []

        self.main_goal = Perception(env.get_accurate_goal_state())

        while not done:
            state = Perception(state)
            assert len(state) * 2 == self.cfg.classifier_length

            match_set = self.population.form_match_set(state)
            action = self.cfg.action_selector(match_set)

            logger.debug("\tExecuting action: [%d]", action)

            prev_state = Perception(state)
            raw_state, last_reward, done, _ = env.step(action)
            state = Perception(raw_state)

            trial_steps.append(
                [prev_state, action, last_reward, state, done])

            self.memory.add(ReplayMemorySample(
                state_goal_concat(prev_state, self.main_goal),
                action,
                last_reward,
                state_goal_concat(state, self.main_goal),
                done))

        if state != self.main_goal:
            for index, step in enumerate(trial_steps):
                state, action, reward, next_state, done = step

                self.additional_goals = self.sample_goals(trial_steps, index)

                for goal in self.additional_goals:
                    new_reward = self.reward_function(next_state, goal)

                    self.memory.add(ReplayMemorySample(
                        state_goal_concat(state, goal),
                        action,
                        new_reward,
                        state_goal_concat(next_state, goal),
                        False))

        for _ in trial_steps:
            self.try_learn(time, len(trial_steps))

        return TrialMetrics(len(trial_steps), last_reward)

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
            match_set = self.population.form_match_set(
                empty_goal_to_state(state, self.cfg.classifier_wildcard))

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

    def try_learn(self, time, steps):
        if len(self.memory) >= self.cfg.er_min_samples:
            experiences = self.memory.sample()
            for exp in experiences:
                er_match_set = self.population.form_match_set(
                    exp.state)
                er_action_set = er_match_set.form_action_set(
                    exp.action)
                er_next_match_set = self.population.form_match_set(
                    exp.next_state)
                # Apply learning in the replied action set
                ClassifiersList.apply_alp(
                    self.population,
                    er_next_match_set,
                    er_action_set,
                    exp.state,
                    exp.action,
                    exp.next_state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                ClassifiersList.apply_reinforcement_learning(
                    er_action_set,
                    exp.reward,
                    0 if exp.done
                    else er_next_match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma
                )
                if self.cfg.do_ga:
                    ClassifiersList.apply_ga(
                        time + steps,
                        self.population,
                        ClassifiersList() if exp.done else er_next_match_set,
                        er_action_set,
                        exp.next_state,
                        self.cfg.theta_ga,
                        self.cfg.mu,
                        self.cfg.chi,
                        self.cfg.theta_as,
                        self.cfg.do_subsumption,
                        self.cfg.theta_exp)

    def sample_goals(self, trial_steps, index):
        if self.cfg.her_goals_number == 1:
            # 'final' strategy
            steps = [trial_steps[-1]]
        else:
            # 'future' strategy
            steps_taken = len(trial_steps)
            k = min(self.cfg.her_goals_number, steps_taken - index)

            steps = random.sample(trial_steps[index:],
                                  k=k) if k > 0 else []

        return [s[-2] for s in steps]

    @staticmethod
    def reward_function(state, new_goal):
        return 0 if state == new_goal else -1
