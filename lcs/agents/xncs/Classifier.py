import logging
from typing import Union, Optional

from lcs.agents.xcs import Condition
from lcs.agents.xncs import Effect, Configuration
import lcs.agents.xcs as xcs


class Classifier(xcs.Classifier):
    def __init__(self,
                 cfg: Optional[Configuration] = None,
                 condition: Union[Condition, str, None] = None,
                 action: Optional[int] = None,
                 time_stamp: int = None,
                 effect:  Union[Effect, str, None] = None) -> None:
        self.effect = effect
        super().__init__(cfg, condition, action, time_stamp)

    def __hash__(self):
       return hash((str(self.condition), self.action, self.effect))

    def __str__(self):
        return f"Cond:{self.condition} - Act:{self.action} - effect:{self.effect} - Num:{self.numerosity} " + \
            f"[fit: {self.fitness:.3f}, exp: {self.experience:3.2f}, pred: {self.prediction:2.3f}]"
