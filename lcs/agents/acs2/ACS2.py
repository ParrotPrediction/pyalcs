import logging
from typing import Optional

from . import ClassifiersList, Configuration
from ...agents import Agent
from ...agents.Agent import Metric


class ACS2(Agent):
    def __init__(self, cfg: Configuration, population=None) -> None:
        self.cfg = cfg
        self.population = population or ClassifiersList(cfg=self.cfg)

    def explore(self, env, trials):
        """
        Explores the environment in given set of trials.
        :param env: environment
        :param trials: number of trials
        :return: population of classifiers and metrics
        """
        return self._evaluate(env, trials, self._run_trial_explore)

    def exploit(self, env, trials):
        """
        Exploits the environments in given set of trials (always executing
        best possible action - no exploration).
        :param env: environment
        :param trials: number of trials
        :return: population of classifiers and metrics
        """
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

    def _evaluate(self, env, max_trials, func):
        """
        Runs the classifier in desired strategy (see `func`) and collects
        metrics.
        :param env: environment
        :param max_trials: number of trials
        :param func: three arguments: env, steps already made, current trial
        :return: population of classifiers and metrics
        """
        current_trial = 0
        steps = 0

        metrics = []
        while current_trial < max_trials:
            steps_in_trial = func(env, steps, current_trial)
            steps += steps_in_trial

            trial_metrics = self._collect_metrics(
                env, current_trial, steps_in_trial, steps)
            logging.info(trial_metrics)
            metrics.append(trial_metrics)

            current_trial += 1

        return self.population, metrics

    def _run_trial_explore(self, env, time, current_trial=None):
        logging.debug("** Running trial explore ** ")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self._parse_state(raw_state)
        action = None
        reward = None
        prev_state = None
        action_set = ClassifiersList(cfg=self.cfg)
        done = False

        while not done:
            match_set = ClassifiersList.form_match_set(self.population,
                                                       state,
                                                       self.cfg)

            if steps > 0:
                # Apply learning in the last action set
                action_set.apply_alp(
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.population,
                    match_set)
                action_set.apply_reinforcement_learning(
                    reward,
                    match_set.get_maximum_fitness())
                if self.cfg.do_ga:
                    action_set.apply_ga(
                        time + steps,
                        self.population,
                        match_set,
                        state)

            action = match_set.choose_action(self.cfg.epsilon)
            logging.debug("\tExecuting action: [%d]", action)
            action_set = ClassifiersList.form_action_set(match_set,
                                                         action,
                                                         self.cfg)

            prev_state = state
            raw_state, reward, done, _ = env.step(self._parse_action(action))
            state = self._parse_state(raw_state)

            if done:
                action_set.apply_alp(
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.population,
                    None)
                action_set.apply_reinforcement_learning(
                    reward,
                    0)
            if self.cfg.do_ga:
                action_set.apply_ga(
                    time + steps,
                    self.population,
                    None,
                    state)

            steps += 1

        return steps

    def _run_trial_exploit(self, env, time=None, current_trial=None):
        logging.debug("** Running trial exploit **")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self._parse_state(raw_state)

        reward = None
        action_set = ClassifiersList(cfg=self.cfg)
        done = False

        while not done:
            match_set = ClassifiersList.form_match_set(self.population,
                                                       state,
                                                       self.cfg)

            if steps > 0:
                action_set.apply_reinforcement_learning(
                    reward,
                    match_set.get_maximum_fitness())

            # Here while exploiting always choose best action
            action = match_set.choose_action(epsilon=0.0)
            action_set = ClassifiersList.form_action_set(match_set,
                                                         action,
                                                         self.cfg)

            raw_state, reward, done, _ = env.step(self._parse_action(action))
            state = self._parse_state(raw_state)

            if done:
                action_set.apply_reinforcement_learning(reward, 0)

            steps += 1

        return steps

    def _parse_state(self, raw_state):
        """
        Sometimes the environment state returned by the OpenAI
        environment does not suit to the classifier representation
        of data used by ACS2. If a mapping function is defined in
        configuration - use it.

        :param raw_state: state obtained from OpenAI gym
        :return: state suitable for ACS2 (list)
        """
        if self.cfg.perception_mapper_fcn:
            return self.cfg.perception_mapper_fcn(raw_state)

        return raw_state

    def _parse_action(self, action_idx):
        """
        Sometimes the step function from OpenAI Gym takes different
        representation of actions than sequential range of integers.
        There is a possiblity to provide custom mapping function for
        suitable action values.

        :param action_idx: action id, used in ACS2
        :return: action id for the step function
        """
        if self.cfg.action_mapping_dict:
            return self.cfg.action_mapping_dict[action_idx]

        return action_idx

    def _collect_agent_metrics(self, trial, steps, total_steps) -> Metric:
        return {
            'population': len(self.population),
            'numerosity': sum(cl.num for cl in self.population),
            'reliable': len([cl for cl in
                             self.population if cl.is_reliable()]),
            'fitness': (sum(cl.fitness for cl in self.population) /
                        len(self.population)),
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
