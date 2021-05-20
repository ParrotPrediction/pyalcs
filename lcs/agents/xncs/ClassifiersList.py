import numpy as np
import logging
import random

import lcs.agents.xcs as xcs
from lcs.agents.xcs import Condition
from lcs.agents.xncs import Classifier, Configuration, Effect
logger = logging.getLogger(__name__)


class ClassifiersList(xcs.ClassifiersList):
    
    def __init__(self, 
                 cfg: Configuration,
                 *args,
                 oktypes=(Classifier,),
                 ) -> None:
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
            effect.append(str(random.randint(0, 1)))
        cl = Classifier(cfg=self.cfg,
                        condition=Condition(generalized),
                        action=action,
                        time_stamp=time_stamp,
                        effect=Effect(effect))
        return cl

    @property
    def fittest_classifier(self):
        return max(self, key=lambda cl: cl.fitness * cl.prediction)
