import logging
import random
from copy import copy
from typing import Optional

import numpy as np

from lcs.agents import Agent
from lcs.agents.Agent import TrialMetrics
from lcs.agents.xcs import Configuration, ClassifiersList
# TODO: delete old code
from lcs.strategies.reinforcement_learning import simple_q_learning
from lcs.agents.xcs import GeneticAlgorithm
logger = logging.getLogger(__name__)


class XCS(Agent):
    def __init__(self,
                 cfg: Configuration,
                 population: Optional[ClassifiersList] = None
                 ) -> None:
        """
        :param cfg: object storing parameters of the experiment
        :param population: all classifiers at current time
        """
        self.cfg = cfg
        if population is not None:
            self.population = population
        else:
            self.population = ClassifiersList(cfg=cfg)
        self.ga = GeneticAlgorithm(
            population=self.population,
            cfg=self.cfg
        )
        self.time_stamp = 0
        self.reward = 0

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        # Doubling _run_trials_explore would cause too many issues.
        temp = self.cfg.epsilon
        self.cfg.epsilon = 0
        metrics = self._run_trial_explore(env, trials, current_trial)
        self.cfg.epsilon = temp
        return metrics

    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        prev_action_set = None
        prev_reward = self.reward
        prev_state = None  # state is known as situation
        prev_time_stamp = self.time_stamp  # steps
        done = False  # eop

        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)

        while not done:
            assert len(self.population) == len(set(self.population)), 'duplicates found'
            self.population.delete_from_population()
            # We are in t+1 here
            match_set = self.population.generate_match_set(state, self.time_stamp)
            prediction_array = match_set.prediction_array
            action = self.select_action(prediction_array, match_set)
            action_set = match_set.generate_action_set(action)
            # apply action to environment
            raw_state, step_reward, done, _ = env.step(action)
            state = self.cfg.environment_adapter.to_genotype(raw_state)
            if self.cfg.multistep_enfiroment:
                self.reward = step_reward + self.cfg.gamma * self.reward

            self._distribute_and_update(prev_action_set,
                                        prev_state,
                                        prev_reward + self.cfg.gamma * max(prediction_array))
            if done:
                self._distribute_and_update(action_set,
                                            state,
                                            self.reward)
            else:
                prev_action_set = copy(action_set)
                prev_reward = self.reward
                prev_state = copy(state)
            self.time_stamp += 1
        return TrialMetrics(self.time_stamp - prev_time_stamp, self.reward)

    def _distribute_and_update(self, action_set, situation, p):
        if action_set is not None and len(action_set) > 0:
            action_set.update_set(p)
            if self.cfg.do_action_set_subsumption:
                self.do_action_set_subsumption(action_set)
            self.ga.run_ga(
                action_set,
                situation,
                self.time_stamp
            )

    # TODO: EpsilonGreedy
    # Run into a lot of issues where in EpsilonGreedy where BestAction was not callable
    # Changing EpsilonGreed to:
    # best = BestAction(all_actions=self.all_actions)
    # return best(population)
    # Fixed the issue but I want to solve it without changes to EpsilonGreedy.py
    def select_action(self, prediction_array, match_set: ClassifiersList) -> int:
        if np.random.rand() > self.cfg.epsilon:
            return max((v, i) for i, v in enumerate(prediction_array))[1]
        return match_set[random.randrange(len(match_set))].action

    def do_action_set_subsumption(self, action_set: ClassifiersList) -> None:
        cl = None
        for c in action_set:
            if c.could_subsume:
                if cl is None or c.more_general(cl):
                    cl = c
        if cl is not None:
            for c in action_set:
                if cl.is_more_general(c):
                    cl.numerosity += c.numerosity
                    action_set.safe_remove(c)
                    self.population.safe_remove(c)
