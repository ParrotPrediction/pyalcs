import logging
from typing import Union, Optional

from lcs.agents.xcs import Condition
from lcs.agents.xcs import Classifier as ClassifierXCS
from lcs.agents.xncs import Effect, Configuration


class Classifier(ClassifierXCS):
    def __init__(self,
                 cfg: Optional[Configuration] = None,
                 condition: Union[Condition, str, None] = None,
                 effect:  Union[Effect, str, None] = None,
                 action: Optional[int] = None,
                 time_stamp: int = None) -> None:
        self.effect = effect
        super().__init__(cfg, condition, action, time_stamp)

