import logging
from collections import namedtuple
from timeit import default_timer as timer
from typing import Callable, List, Tuple

import numpy as np

from lcs.metrics import basic_metrics

TrialMetrics = namedtuple('TrialMetrics', ['steps', 'reward'])

logger = logging.getLogger(__name__)


class Agent:

    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        raise NotImplementedError()

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        raise NotImplementedError()

    def get_population(self):
        raise NotImplementedError()

    def get_cfg(self):
        raise NotImplementedError()

    def explore(self, env, trials, decay: bool = False) -> Tuple:
        """
        Explores the environment in given set of trials.

        Parameters
        ----------
        env
            environment
        trials
            number of trials
        decay
            whether the epsilon is decaying along trials

        Returns
        -------
        Tuple
            population of classifiers and metrics
        """
        return self._evaluate(env, trials, self._run_trial_explore, decay)

    def exploit(self, env, trials) -> Tuple:
        """
        Exploits the environments in given set of trials (always executing
        best possible action - no exploration).

        Parameters
        ----------
        env
            environment
        trials
            number of trials

        Returns
        -------
        Tuple
            population of classifiers and metrics
        """
        return self._evaluate(env, trials, self._run_trial_exploit)

    def explore_exploit(self, env, trials) -> Tuple:
        """
        Alternates between exploration and exploitation phases.

        Parameters
        ----------
        env
            environment
        trials
            number of trials

        Returns
        -------
        Tuple
            population of classifiers and metrics
        """

        def switch_phases(env, steps, current_trial):
            if current_trial % 2 == 0:
                return self._run_trial_explore(env, steps, current_trial)
            else:
                return self._run_trial_exploit(env, None, current_trial)

        return self._evaluate(env, trials, switch_phases)

    def _evaluate(self,
                  env,
                  n_trials: int,
                  func: Callable,
                  decay: bool = False) -> Tuple:
        """
        Runs the classifier in desired strategy (see `func`) and collects
        metrics.

        Parameters
        ----------
        env:
            OpenAI Gym environment
        n_trials: int
            maximum number of trials
        func: Callable
            Function accepting three parameters: env, steps already made,
             current trial
        decay: bool
            Whether the epsilon is decaying through the whole experiment

        Returns
        -------
        tuple
            population of classifiers and metrics
        """
        current_trial = 0
        steps = 0

        metrics: List = []
        while current_trial < n_trials:
            start_ts = timer()
            steps_in_trial, reward = func(env, steps, current_trial)
            end_ts = timer()

            steps += steps_in_trial

            # collect user metrics
            if current_trial % self.get_cfg().metrics_trial_frequency == 0:
                m = basic_metrics(
                    current_trial, steps_in_trial, reward, end_ts - start_ts)

                user_metrics = self.get_cfg().user_metrics_collector_fcn
                if user_metrics is not None:
                    m.update(user_metrics(self, env))

                metrics.append(m)

            # Print last metric
            if current_trial % np.round(n_trials / 10) == 0:
                logger.info(metrics[-1])

            if decay:
                # Gradually decrease the epsilon
                self.get_cfg().epsilon -= 1 / n_trials
                if self.get_cfg().epsilon < 0.01:
                    self.get_cfg().epsilon = 0.01

            current_trial += 1

        return self.get_population(), metrics
