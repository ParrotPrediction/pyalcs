from __future__ import annotations
from lcs import Perception
from dataclasses import dataclass

@dataclass
class ReplayMemorySample:
    state: Perception
    action: int
    reward: float
    next_state: Perception
    done: bool
