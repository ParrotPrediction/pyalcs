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
            cfg=self.cfg
            )
        self.time_stamp = 0
        self.reward = 0
        self.mistakes = []

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
