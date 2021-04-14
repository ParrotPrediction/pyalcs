from typing import Optional
import random
import numpy as np
from copy import copy

from lcs.agents.xcs import XCS
from lcs.agents.xncs import Configuration, Backpropagation
# TODO: find typo that makes __init__ not do that
from lcs.agents.xncs.ClassifiersList import ClassifiersList
from lcs.agents.Agent import TrialMetrics
from lcs.strategies.reinforcement_learning import simple_q_learning


class XNCS(XCS):

    def __init__(self,
                 cfg: Configuration,
                 population: Optional[ClassifiersList] = None
                 ) -> None:
        """
        :param cfg: object storing parameters of the experiment
        :param population: all classifiers at current time
        """
        self.back_propagation = Backpropagation(cfg)
        self.cfg = cfg
        if population is not None:
            self.population = population
        else:
            self.population = ClassifiersList(cfg=cfg)
        self.time_stamp = 0
        self.action_reward = [0 for _ in range(cfg.number_of_actions)]

    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        prev_action_set = None
        prev_reward = [0 for _ in range(self.cfg.number_of_actions)]
        prev_state = None  # state is known as situation
        prev_action = 0
        self.time_stamp = 0  # steps
        done = False  # eop

        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)

        while not done:
            self.population.delete_from_population()
            # We are in t+1 here
            match_set = self.population.generate_match_set(state, self.time_stamp)
            prediction_array = match_set.prediction_array
            action = self.select_action(prediction_array, match_set)
            action_set = match_set.generate_action_set(action)
            # apply action to environment
            raw_state, step_reward, done, _ = env.step(action)
            state = self.cfg.environment_adapter.to_genotype(raw_state)
            self.action_reward[action] = simple_q_learning(self.action_reward[action],
                                                           step_reward,
                                                           self.cfg.learning_rate,
                                                           self.cfg.gamma,
                                                           match_set.best_prediction)

            self._distribute_and_update(prev_action_set,
                                        prev_state,
                                        prev_reward[prev_action] + self.cfg.gamma * max(prediction_array))
            if done:
                self._distribute_and_update(action_set,
                                            state,
                                            self.action_reward[action])
            else:
                prev_action_set = copy(action_set)
                prev_reward[action] = copy(self.action_reward[action])
                prev_state = copy(state)
                prev_action = action

            self.time_stamp += 1
        return TrialMetrics(self.time_stamp, self.action_reward)

    def _distribute_and_update(self, action_set, situation, p):
        super()._distribute_and_update(action_set, situation, p)
        self._compare_effect(action_set, situation)

    def _compare_effect(self, action_set, situation):
        if action_set is not None:
            for cl in action_set:
                if cl.effect is None or not cl.effect.subsumes(situation):
                    self.back_propagation.insert_into_bp(cl, situation)
                else:
                    self.back_propagation.update_bp()
            self.back_propagation.check_and_update()
