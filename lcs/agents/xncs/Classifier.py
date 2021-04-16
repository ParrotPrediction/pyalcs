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

    def __eq__(self, other):
        if other.action == self.action \
           and other.condition == self.condition:
            if other.effect is None and self.effect is None:
                return True
            if other.effect is None:
                return False
            if self.effect is None:
                return False
            if other.effect == self.effect:
                return True
        return False
