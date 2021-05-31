import numpy as np
import logging
import random

import lcs.agents.xcs as xcs
from lcs import Perception
from lcs.agents.xcs import Condition
from lcs.agents.xncs import Classifier, Configuration, Effect, Backpropagation
logger = logging.getLogger(__name__)


class ClassifiersList(xcs.ClassifiersList):
    
    def __init__(self, 
                 cfg: Configuration,
                 *args,
                 oktypes=(Classifier,),
                 ) -> None:
        self.back_propagation = Backpropagation(
            cfg=cfg,
            percentage=0.2
            )
        super().__init__(cfg, *args, oktypes=oktypes)

    # without this function the Classifierlist will create XCS Classifiers
    # instead of XNCS classifiers
    def generate_covering_classifier(self, situation, action, time_stamp):
        generalized = []
        effect = []
        for i in range(len(situation)):
            if np.random.rand() > self.cfg.covering_wildcard_chance:
                generalized.append(self.cfg.classifier_wildcard)
            else:
                generalized.append(situation[i])
            effect.append(str(random.choice(situation)))
        cl = Classifier(cfg=self.cfg,
                        condition=Condition(generalized),
                        action=action,
                        time_stamp=time_stamp,
                        effect=Effect(effect))
        return cl

    def generate_action_set(self, action):
        action_ls = [cl for cl in self if cl.action == action]
        return ClassifiersList(self.cfg, *action_ls)

    def generate_match_set(self, situation: Perception, time_stamp):
        matching_ls = [cl for cl in self if cl.does_match(situation)]
        action = self._find_not_present_action(matching_ls)
        while action is not None:
            cl = self._generate_covering_and_insert(situation, action, time_stamp)
            matching_ls.append(cl)
            action = self._find_not_present_action(matching_ls)
        return ClassifiersList(self.cfg, *matching_ls)

    def least_fit_classifiers(self, percentage):
        assert 0 < percentage <= 1
        return sorted(
                    self,
                    key=lambda cl: cl.fitness
               )[0:int(len(self) * percentage)]

    @property
    def fittest_classifier(self):
        assert len(self) > 0
        return max(self, key=lambda cl: cl.fitness * cl.prediction)
