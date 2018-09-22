import logging
from typing import Optional, Callable, Tuple, List

from lcs.strategies.action_selection import choose_action
from ...agents import Agent
from ...agents.Agent import Metric
from ...agents.racs import Configuration, ClassifierList
from ...utils import parse_state, parse_action


logger = logging.getLogger(__name__)


class RACS(Agent):
    """ACS2 agent operating on real-valued (floating) number"""

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifierList=None) -> None:
        self.cfg = cfg
        self.population = population or ClassifierList()

    def explore(self, env, trials) -> Tuple:
        return self._evaluate(env, trials, self._run_trial_explore)

    def exploit(self, env, trials):
        pass

    def _evaluate(self, env, max_trials: int, func: Callable) -> Tuple:
        """
        Runs the classifier in desired strategy (see `func`) and collects
        metrics.

        Parameters
        ----------
        env:
            OpenAI Gym environment
        max_trials: int
            maximum number of trials
        func: Callable
            Function accepting three parameters: env, steps already made,
             current trial

        Returns
        -------
        tuple
            population of classifiers and metrics
        """
        current_trial = 0
        steps = 0

        metrics: List = []
        while current_trial < max_trials:
            steps_in_trial = func(env, steps, current_trial)
            steps += steps_in_trial

            # TODO: collect metrics

            current_trial += 1

        return self.population, metrics

    def _run_trial_explore(self, env, time, current_trial=None):
        logger.debug("** Running trial explore ** ")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = parse_state(raw_state, self.cfg.perception_mapper_fcn)
        action = None
        reward = None
        prev_state = None
        action_set = ClassifierList()
        done = False

        while not done:
            match_set = self.population.form_match_set(state)

            if steps > 0:
                # Apply learning in the last action set
                action_set.apply_alp(
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.population,
                    match_set,
                    self.cfg)
                action_set.apply_reinforcement_learning(
                    reward,
                    match_set.get_maximum_fitness())
                if self.cfg.do_ga:
                    pass
                    # TODO: implement GA

            action = choose_action(
                match_set,
                self.cfg.number_of_possible_actions,
                self.cfg.epsilon)
            internal_action = parse_action(action, self.cfg.action_mapping_fcn)
            logger.debug("\tExecuting action: [%d]", action)
            action_set = match_set.form_action_set(action)

            prev_state = state
            raw_state, reward, done, _ = env.step(internal_action)
            state = parse_state(raw_state, self.cfg.perception_mapper_fcn)

            if done:
                action_set.apply_alp(
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.population,
                    None,
                    self.cfg)
                action_set.apply_reinforcement_learning(
                    reward,
                    0)
                if self.cfg.do_ga:
                    pass
                    # TODO: implement GA
            steps += 1

        return steps

    def _collect_agent_metrics(self, trial, steps, total_steps) -> Metric:
        return {
            "population": 0
        }

    def _collect_environment_metrics(self, env) -> Optional[Metric]:
        return None

    def _collect_performance_metrics(self, env) -> Optional[Metric]:
        return None
