from __future__ import annotations
from lcs import Perception


class ReplayMemorySample:
    __slots__ = ['state', 'action', 'reward', 'next_state', 'done']

    def __init__(self,
                 state: Perception,
                 action: int,
                 reward: float,
                 next_state: Perception,
                 done: bool) -> None:

        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
