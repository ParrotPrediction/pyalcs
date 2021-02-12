from dataclasses import dataclass
from typing import Union, Optional, Generator, List, Dict
from lcs import TypedList, Perception

from lcs.agents import Agent
from lcs.agents.xcs import Configuration
from lcs.agents.Agent import TrialMetrics
from lcs.agents.ImmutableSequence import ImmutableSequence

from lcs.agents.xcs import Configuration


class Condition:
    def __init__(self):
        raise NotImplementedError()


class Effect:
    def __init__(self):
        raise NotImplementedError()


class Classifier:

    def __init__(self,
                 condition: Union[Condition, str, None] = None,
                 action: Optional[int] = None,
                 effect: Union[Effect, str, None] = None,
                 time_stamp: int = None,
                 cfg: Optional[Configuration] = None) -> None:
        if cfg is None:
            raise TypeError("Configuration should be passed to Classifier")
        self.cfg = cfg

        self.condition = condition
        self.action = action
        self.effect = effect
        self.time_stamp = time_stamp
        self.experience = 0
        self.action_set_size = 1
        self.numerosity = 1
        self.prediction, self.error, self.fitness = cfg.initial_classifier_values()

