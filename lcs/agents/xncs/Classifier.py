import logging
from typing import Union, Optional

from lcs.agents.xcs import Configuration, Condition
from lcs.agents.xcs import Classifier as cl_xcs
from lcs.agents.xncs import Effect


class Classifier(cl_xcs):
    def __init__(self,
                 cfg: Optional[Configuration] = None,
                 condition: Union[Condition, str, None] = None,
                 effect: Union[Effect, str, None] = None,
                 action: Optional[int] = None,
                 time_stamp: int = None) -> None:
        self.effect = effect
        args = (cfg, condition, action, time_stamp)
        super.__init__(*args)

