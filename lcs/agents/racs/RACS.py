import logging
from typing import Optional, Callable, Tuple, List, Dict, Any

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
        return self._evaluate(env, trials, self._run_trial_exploit)

    def explore_exploit(self, env, trials):
        """
        Alternates between exploration and exploitation phases.
        :param env: environment
        :param trials: number of trials
        :return: population of classifiers and metrics
        """
        def switch_phases(env, steps, current_trial):
            if current_trial % 2 == 0:
                return self._run_trial_explore(env, steps)
            else:
                return self._run_trial_exploit(env, None)

        return self._evaluate(env, trials, switch_phases)

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
            steps_in_trial, reward = func(env, steps, current_trial)
            steps += steps_in_trial

            trial_metrics = self._collect_metrics(
                env, current_trial, steps_in_trial, steps, reward)
            metrics.append(trial_metrics)

            if current_trial % 1000 == 0:
                logger.info(trial_metrics)

            current_trial += 1

        return self.population, metrics

    def _run_trial_explore(self, env, time, current_trial=None):
        """
        Executes explore trial

        Parameters
        ----------
        env
        time
        current_trial

        Returns
        -------
        Tuple[int, int]
            Tuple of total steps taken and final reward
        """
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
                self.cfg.epsilon)
            internal_action = parse_action(action, self.cfg.action_mapping_fcn)
            logger.debug("\tExecuting action: [%d]", action)
            action_set = match_set.form_action_set(action)

            prev_state = state
            raw_state, reward, done, _ = env.step(internal_action)
            state = parse_state(raw_state, self.cfg.perception_mapper_fcn)

            if done:
                ClassifierList.apply_alp(
                    self.population,
                    None,
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

        return steps, reward

    def _run_trial_exploit(self, env, time=None, current_trial=None):
        logger.debug("** Running trial exploit **")

        steps = 0
        raw_state = env.reset()
        state = parse_state(raw_state, self.cfg.perception_mapper_fcn)

        reward = None
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
                epsilon=0.0)
            internal_action = parse_action(action, self.cfg.action_mapping_fcn)
            action_set = match_set.form_action_set(action)

            raw_state, reward, done, _ = env.step(internal_action)
            state = parse_state(raw_state, self.cfg.perception_mapper_fcn)

            if done:
                ClassifierList.apply_reinforcement_learning(
                    action_set,
                    reward,
                    0,
                    self.cfg.beta,
                    self.cfg.gamma)

            steps += 1

        return steps, reward

    def _collect_agent_metrics(self, trial, steps, total_steps) -> Metric:
        regions = self._count_averaged_regions()

        return {
            'population': len(self.population),
            'numerosity': sum(cl.num for cl in self.population),
            'reliable': len([cl for cl in
                             self.population if cl.is_reliable()]),
            'fitness': (sum(cl.fitness for cl in self.population) /
                        len(self.population)),
            'cover_ratio': (sum(cl.condition.cover_ratio for cl
                                in self.population) / len(self.population)),
            'region_1': regions[1],
            'region_2': regions[2],
            'region_3': regions[3],
            'region_4': regions[4],
            'trial': trial,
            'steps': steps,
            'total_steps': total_steps
        }

    def _collect_environment_metrics(self, env) -> Optional[Metric]:
        if self.cfg.environment_metrics_fcn:
            return self.cfg.environment_metrics_fcn(env)

        return None

    def _collect_performance_metrics(self, env, reward) -> Optional[Metric]:
        basic_metrics = {
            'reward': reward
        }

        extra_metrics: Dict[str, Any] = {}

        if self.cfg.performance_fcn:
            extra_metrics = self.cfg.performance_fcn(
                env, self.population, **self.cfg.performance_fcn_params)

        return {**basic_metrics, **extra_metrics}

    def _count_averaged_regions(self) -> Dict[int, float]:
        region_counts = {1: 0, 2: 0, 3: 0, 4: 0}

        for cl in self.population:
            for region, counts in cl.get_interval_proportions().items():
                region_counts[region] += counts

        all_elems = sum(i for r, i in region_counts.items())

        return {r: i / all_elems for r, i in region_counts.items()}
