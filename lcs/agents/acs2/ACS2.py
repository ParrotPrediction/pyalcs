import logging
from typing import Tuple

from lcs import Perception
from lcs.agents.Agent import TrialMetrics
from lcs.strategies.action_planning.action_planning import \
    search_goal_sequence, exists_classifier
from . import ClassifiersList, Configuration
from ...agents import Agent
from ...strategies.action_selection import choose_action

logger = logging.getLogger(__name__)


class ACS2(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList=None) -> None:
        self.cfg = cfg
        self.population = population or ClassifiersList()

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, time, current_trial=None) \
            -> TrialMetrics:

        logger.debug("** Running trial explore ** ")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = Perception(self.cfg.environment_adapter.to_genotype(raw_state))
        action = env.action_space.sample()
        reward = 0
        prev_state = Perception.empty()
        action_set = ClassifiersList()
        done = False

        while not done:
            if self.cfg.do_action_planning and \
                    self._time_for_action_planning(steps + time):
                # Action Planning for increased model learning
                steps_ap, state, prev_state, action_set, reward = \
                    self._run_action_planning(env, steps + time, state,
                                              prev_state, action_set, action,
                                              reward)
                steps += steps_ap

            match_set = self.population.form_match_set(state)

            if steps > 0:
                # Apply learning in the last action set
                ClassifiersList.apply_alp(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma
                )
                if self.cfg.do_ga:
                    ClassifiersList.apply_ga(
                        time + steps,
                        self.population,
                        match_set,
                        action_set,
                        state,
                        self.cfg.theta_ga,
                        self.cfg.mu,
                        self.cfg.chi,
                        self.cfg.theta_as,
                        self.cfg.do_subsumption,
                        self.cfg.theta_exp)

            action = choose_action(
                match_set,
                self.cfg.number_of_possible_actions,
                self.cfg.epsilon)
            iaction = self.cfg.environment_adapter.to_lcs_action(action)
            logger.debug("\tExecuting action: [%d]", action)
            action_set = match_set.form_action_set(action)

            prev_state = state
            raw_state, reward, done, _ = env.step(iaction)
            state = Perception(self.cfg.environment_adapter.to_genotype(raw_state))

            if done:
                ClassifiersList.apply_alp(
                    self.population,
                    ClassifiersList(),
                    action_set,
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    reward,
                    0,
                    self.cfg.beta,
                    self.cfg.gamma)
            if self.cfg.do_ga:
                ClassifiersList.apply_ga(
                    time + steps,
                    self.population,
                    ClassifiersList(),
                    action_set,
                    state,
                    self.cfg.theta_ga,
                    self.cfg.mu,
                    self.cfg.chi,
                    self.cfg.theta_as,
                    self.cfg.do_subsumption,
                    self.cfg.theta_exp)

            steps += 1

        return TrialMetrics(steps, reward)

    def _run_trial_exploit(self, env, time=None, current_trial=None) \
            -> TrialMetrics:

        logger.debug("** Running trial exploit **")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)

        reward = 0
        action_set = ClassifiersList()
        done = False

        while not done:
            match_set = self.population.form_match_set(state)

            if steps > 0:
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma)

            # Here when exploiting always choose best action
            action = choose_action(
                match_set,
                self.cfg.number_of_possible_actions,
                epsilon=0.0)
            iaction = self.cfg.environment_adapter.to_lcs_action(action)
            action_set = match_set.form_action_set(action)

            raw_state, reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(raw_state)

            if done:
                ClassifiersList.apply_reinforcement_learning(
                    action_set, reward, 0, self.cfg.beta, self.cfg.gamma)

            steps += 1

        return TrialMetrics(steps, reward)

    def _run_action_planning(self,
                             env,
                             time: int,
                             state: Perception,
                             prev_state: Perception,
                             action_set: ClassifiersList,
                             action: int,
                             reward: int) -> Tuple[int, Perception, Perception,
                                                   ClassifiersList, int]:
        """
        Executes action planning for model learning speed up.
        Method requests goals from 'goal generator' provided by
        the environment. If goal is provided, ACS2 searches for
        a goal sequence in the current model (only the reliable classifiers).
        This is done as long as goals are provided and ACS2 finds a sequence
        and successfully reaches the goal.

        Parameters
        ----------
        env
        time
        state
        prev_state
        action_set
        action
        reward

        Returns
        -------

        """
        logging.debug("** Running action planning **")

        if not hasattr(env.env, "get_goal_state"):
            logging.debug("Action planning stopped - "
                          "no function get_goal_state in env")
            return 0, state, prev_state, action_set, reward

        steps = 0
        done = False

        while not done:
            goal_situation = env.env.get_goal_state()

            if goal_situation is None:
                break

            act_sequence = search_goal_sequence(self.population, state,
                                                goal_situation,
                                                self.cfg.theta_r)

            # Execute the found sequence and learn during executing
            i = 0
            for act in act_sequence:
                if act == -1:
                    break

                match_set = self.population.form_match_set(state)

                if action_set is not None and len(prev_state) != 0:
                    ClassifiersList.apply_alp(
                        self.population,
                        match_set,
                        action_set,
                        prev_state,
                        action,
                        state,
                        time + steps,
                        self.cfg.theta_exp,
                        self.cfg)
                    ClassifiersList.apply_reinforcement_learning(
                        action_set,
                        reward,
                        0,
                        self.cfg.beta,
                        self.cfg.gamma)
                    if self.cfg.do_ga:
                        ClassifiersList.apply_ga(
                            time + steps,
                            self.population,
                            match_set,
                            action_set,
                            state,
                            self.cfg.theta_ga,
                            self.cfg.mu,
                            self.cfg.chi,
                            self.cfg.theta_as,
                            self.cfg.do_subsumption,
                            self.cfg.theta_exp)

                action = act
                action_set = ClassifiersList.form_action_set(match_set, action)

                iaction = self.cfg.environment_adapter.to_lcs_action(action)

                raw_state, reward, done, _ = env.step(iaction)
                prev_state = state

                state = self.cfg.environment_adapter.to_genotype(raw_state)

                if not exists_classifier(action_set, prev_state,
                                         action, state,
                                         self.cfg.theta_r):

                    # no reliable classifier was able to anticipate
                    # such a change
                    break

                steps += 1
                i += 1

            if i == 0:
                break

        return steps, state, prev_state, action_set, reward

    def _time_for_action_planning(self, time):
        return time % self.cfg.action_planning_frequency == 0
