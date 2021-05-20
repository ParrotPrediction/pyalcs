from typing import Optional
import random
import numpy as np
from copy import copy

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
            population=self.population
            )
        self.time_stamp = 0
        self.reward = 0

    def _distribute_and_update(self, action_set, situation, p):
        super()._distribute_and_update(action_set, situation, p)
        self._compare_effect(action_set, situation)

    def _compare_effect(self, action_set: ClassifiersList, situation):
        if action_set is not None:
            self.back_propagation.insert_into_bp(
                action_set.fittest_classifier,
                Effect(situation)
            )
