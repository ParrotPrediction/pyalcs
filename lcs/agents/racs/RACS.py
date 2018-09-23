import logging
from typing import Optional, Callable, Tuple, List

from lcs.strategies.action_selection import choose_action
from ...agents import Agent
from ...agents.Agent import Metric
from ...agents.racs import Configuration, ClassifierList
from ...utils import parse_state, parse_action


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

            trial_metrics = self._collect_metrics(
                env, current_trial, steps_in_trial, steps)
            metrics.append(trial_metrics)

            if current_trial % 25 == 0:
                logging.info(trial_metrics)

            current_trial += 1

        return self.population, metrics

    def _run_trial_explore(self, env, time, current_trial=None):
        logging.debug("** Running trial explore ** ")

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
            logging.debug("\tExecuting action: [%d]", action)
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

        return steps

    def _run_trial_exploit(self, env, time=None, current_trial=None):
        logging.debug("** Running trial exploit **")

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

        return steps

    def _collect_agent_metrics(self, trial, steps, total_steps) -> Metric:
        return {
            'population': len(self.population),
            'numerosity': sum(cl.num for cl in self.population),
            'reliable': len([cl for cl in
                             self.population if cl.is_reliable()]),
            'fitness': (sum(cl.fitness for cl in self.population) /
                        len(self.population)),
            'cover_ratio': (sum(cl.condition.cover_ratio for cl
                                in self.population) / len(self.population)),
            'trial': trial,
            'steps': steps,
            'total_steps': total_steps
        }

    def _collect_environment_metrics(self, env) -> Optional[Metric]:
        if self.cfg.environment_metrics_fcn:
            return self.cfg.environment_metrics_fcn(env)

        return None

    def _collect_performance_metrics(self, env) -> Optional[Metric]:
        if self.cfg.performance_fcn:
            return self.cfg.performance_fcn(
                env, self.population, **self.cfg.performance_fcn_params)

        return None
