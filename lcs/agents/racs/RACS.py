import logging

from lcs import Perception
from lcs.agents.Agent import TrialMetrics
from lcs.agents.racs.action_selection import choose_action
from ...agents import Agent
from ...agents.racs import Configuration, ClassifierList

logger = logging.getLogger(__name__)


class RACS(Agent):
    """ACS2 agent operating on real-valued (floating) number"""

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifierList = None) -> None:
        self.cfg = cfg
        self.population = population or ClassifierList()

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, time, current_trial=None) \
            -> TrialMetrics:
        """
        Executes explore trial

        Parameters
        ----------
        env
        time

        Returns
        -------
        Tuple[int, int]
            Tuple of total steps taken and final reward
        """
        logger.debug("** Running trial explore ** ")

        # Initial conditions
        steps = 0
        state = env.reset()

        action = env.action_space.sample()
        reward = 0
        prev_state = Perception.empty()
        action_set = ClassifierList()
        done = False

        while not done:
            match_set = self.population.form_match_set(state)

            if steps > 0:
                # Apply learning in the last action set
                ClassifierList.apply_alp(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                ClassifierList.apply_reinforcement_learning(
                    action_set,
                    reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma)
                if self.cfg.do_ga:
                    ClassifierList.apply_ga(
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
                self.cfg.epsilon,
                self.cfg.biased_exploration
            )
            logger.debug("\tExecuting action: [%d]", action)
            action_set = match_set.form_action_set(action)

            prev_state = state
            raw_state, reward, done, _ = env.step(action)
            state = raw_state

            if done:
                ClassifierList.apply_alp(
                    self.population,
                    ClassifierList(),
                    action_set,
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                ClassifierList.apply_reinforcement_learning(
                    action_set,
                    reward,
                    0,
                    self.cfg.beta,
                    self.cfg.gamma)
                if self.cfg.do_ga:
                    ClassifierList.apply_ga(
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
            steps += 1

        return TrialMetrics(steps, reward)

    def _run_trial_exploit(self, env, time=None, current_trial=None) \
            -> TrialMetrics:
        logger.debug("** Running trial exploit **")

        steps = 0
        state = env.reset()

        reward = 0
        action_set = ClassifierList()
        done = False

        while not done:
            match_set = self.population.form_match_set(state)

            if steps > 0:
                ClassifierList.apply_reinforcement_learning(
                    action_set,
                    reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma)

            # Execute best action
            action = choose_action(
                match_set,
                self.cfg.number_of_possible_actions,
                epsilon=0.0,
                biased_exploration_prob=0.0)
            action_set = match_set.form_action_set(action)

            state, reward, done, _ = env.step(action)

            if done:
                ClassifierList.apply_reinforcement_learning(
                    action_set,
                    reward,
                    0,
                    self.cfg.beta,
                    self.cfg.gamma)

            steps += 1

        return TrialMetrics(steps, reward)
