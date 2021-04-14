import numpy as np
import logging

import lcs.agents.xcs as xcs
from lcs.agents.xcs import Condition
from lcs.agents.xncs import Classifier, Configuration
logger = logging.getLogger(__name__)


class ClassifiersList(xcs.ClassifiersList):
    
    def __init__(self, 
                 cfg: Configuration,
                 *args,
                 oktypes=(Classifier,),
                 ) -> None:
        super().__init__(cfg, *args, oktypes=oktypes)

    def generate_covering_classifier(self, situation, action, time_stamp):
        generalized = []
        for i in range(len(situation)):
            if np.random.rand() > self.cfg.covering_wildcard_chance:
                generalized.append(self.cfg.classifier_wildcard)
            else:
                generalized.append(situation[i])
        cl = Classifier(cfg=self.cfg,
                        condition=Condition(generalized),
                        action=action,
                        time_stamp=time_stamp)
        return cl
