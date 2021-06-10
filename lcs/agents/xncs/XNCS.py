from typing import Optional
import random
import numpy as np
from copy import copy
import queue
from lcs.agents.xcs import XCS
from lcs.agents.Agent import TrialMetrics
from lcs.agents.xncs import Configuration, Backpropagation
# TODO: find a way to not require super in __init__
from lcs.agents.xncs import ClassifiersList, GeneticAlgorithm, Effect


class XNCS(XCS):

    def __init__(self,
                 cfg: Configuration,
                 population: Optional[ClassifiersList] = None
                 ) -> None:
        """
        :param cfg: object storing parameters of the experiment
        :param population: all classifiers at current time
        """

        if population is not None:
            self.population = population
        else:
            self.population = ClassifiersList(cfg=cfg)
        self.cfg = cfg
        self.ga = GeneticAlgorithm(
            population=self.population,
            cfg=self.cfg
        )
        self.back_propagation = Backpropagation(
            cfg=self.cfg,
            percentage=self.cfg.update_percentage
            )
        self.time_stamp = 0
        self.reward = 0
        self.mistakes = []

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
            action_set, prediction_array, action, match_set = self._form_sets_and_choose_action(state)
            # apply action to environment
            raw_state, step_reward, done, _ = env.step(action)
            state = self.cfg.environment_adapter.to_genotype(raw_state)

            if self.cfg.multistep_enfiroment:
                self.reward = step_reward + self.cfg.gamma * self.reward

            self._distribute_and_update(prev_action_set,
                                        prev_state,
                                        state,
                                        prev_reward + self.cfg.gamma * max(prediction_array))
            if done:
                self._distribute_and_update(action_set,
                                            state,
                                            state,
                                            self.reward)
            else:
                prev_action_set = copy(action_set)
                prev_reward = self.reward
                prev_state = copy(state)
            self.time_stamp += 1
        return TrialMetrics(self.time_stamp - prev_time_stamp, self.reward)

    def _form_sets_and_choose_action(self, state):
        match_set = self.population.generate_match_set(state, self.time_stamp)
        prediction_array = match_set.prediction_array
        action = self.select_action(prediction_array, match_set)
        action_set = match_set.generate_action_set(action)
        return action_set, prediction_array, action, match_set

    def _distribute_and_update(self, action_set, current_situation, next_situation, p):
        if action_set is not None:
            for cl in action_set:
                if cl.effect is None:
                    cl.effect = Effect(next_situation)
            self.update_fraction_accuracy(action_set, next_situation)
            if self.cfg.update_env_input:
                self.back_propagation.update_effect(action_set, next_situation)
            else:
                self.back_propagation.update_effect(action_set, action_set.fittest_classifier.effect)
            self.back_propagation.run_bp(
                action_set,
                Effect(next_situation)
            )
        super()._distribute_and_update(action_set, current_situation, next_situation, p)

    def update_fraction_accuracy(self, action_set, next_vector):
        most_numerous = sorted(action_set, key=lambda cl: -1 * cl.numerosity)[0]
        if most_numerous.effect != Effect(next_vector):
            if len(self.mistakes) >= 100:
                self.mistakes.pop(0)
                self.mistakes.append(1)
            else:
                self.mistakes.append(1)
        else:
            if len(self.mistakes) >= 100:
                self.mistakes.pop(0)
                self.mistakes.append(0)
            else:
                self.mistakes.append(0)

    @property
    def fraction_accuracy(self):
        if len(self.mistakes) > 0:
            return sum(self.mistakes) / len(self.mistakes)
        else:
            return 0
