from __future__ import annotations

import random

from lcs import TypedList
from lcs.agents.acs2er.ReplayMemorySample import ReplayMemorySample


class ReplayBuffer(TypedList):
    """
    Represents the replay memory buffer
    """

    def __init__(self, *args, max_size: int, samples_number: int, oktypes=(ReplayMemorySample,)) -> None:
        super().__init__(*args, oktypes=oktypes)
        self.max_size = max_size
        self.batch_size = samples_number

    def add(self, sample: ReplayMemorySample) -> None:
        if len(self) >= self.max_size:
            self.pop(0)
        self.append(sample)

    def sample(self):
        return random.sample(self, k=self.batch_size)
