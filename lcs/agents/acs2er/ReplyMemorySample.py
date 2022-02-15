from __future__ import annotations

import logging
import random
from typing import Optional, Union, Callable, List

import lcs.agents.acs as acs
from lcs import Perception
from . import Configuration, Effect
from . import ProbabilityEnhancedAttribute

logger = logging.getLogger(__name__)


class ReplyMemorySample:
    __slots__ = ['state', 'action', 'reward', 'next_state']

    def __init__(self,
                 state: Perception,
                 action: int,
                 reward: float,
                 next_state: Perception) -> None:

        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
