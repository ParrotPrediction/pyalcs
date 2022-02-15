from __future__ import annotations

from typing import Optional

from lcs import TypedList, Perception
from lcs.agents.acs import Classifier
from lcs.agents.acs2er.ReplyMemorySample import ReplyMemorySample


class ReplayMemory(TypedList):
    """
    Represents the replay memory buffer
    """

    def __init__(self, *args, oktypes=(ReplyMemorySample,)) -> None:
        super().__init__(*args, oktypes=oktypes)
