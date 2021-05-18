from typing import Optional
import random
import numpy as np
from copy import copy

from lcs.agents.xcs import XCS
from lcs.agents.xncs import Configuration, Backpropagation
# TODO: find a way to not require super in __init__
from lcs.agents.xncs import ClassifiersList, GeneticAlgorithm


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
        if population is not None:
            self.population = population
        else:
            self.population = ClassifiersList(cfg=cfg)
        self.cfg = cfg
        self.ga = GeneticAlgorithm(
            population=self.population,
            cfg=self.cfg
        )
        self.time_stamp = 0
        self.reward = 0

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
