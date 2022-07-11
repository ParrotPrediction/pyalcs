from __future__ import annotations

from typing import Optional, Union

import lcs.agents.acs2 as acs
from lcs.agents.acs import PMark
from lcs import Perception
from .Condition import Condition
from . import Configuration, Effect


class Classifier(acs.Classifier):
    __slots__ = ['condition', 'action', 'effect', 'mark', 'q', 'r',
                 'ir', 'num', 'exp', 'talp', 'tga', 'tav', 'ee', 'cfg']

    def __init__(self,
                 condition: Union[Condition, str, None] = None,
                 action: Optional[int] = None,
                 effect: Union[Effect, str, None] = None,
                 quality: float = None,
                 reward: float = 0.5,
                 immediate_reward: float = 0.0,
                 numerosity: int = 1,
                 experience: int = 1,
                 talp=None,
                 tga: int = 0,
                 tav: float = 0.0,
                 cfg: Optional[Configuration] = None) -> None:

        if cfg is None:
            raise TypeError("Configuration should be passed to Classifier")

        self.cfg = cfg

        def build_perception_string(cls, initial,
                                    length=self.cfg.classifier_length):
            if initial:
                return cls(initial)

            return cls.empty(length=length)

        self.condition = build_perception_string(Condition, condition)
        self.action = action
        self.effect = build_perception_string(Effect, effect)

        self.mark = PMark(cfg=self.cfg)
        if quality is None:
            self.q = self.cfg.initial_q
        else:
            self.q = quality

        self.r = reward
        self.ir = immediate_reward
        self.num = numerosity
        self.exp = experience
        self.talp = talp
        self.tga = tga
        self.tav = tav
        self.ee = False

    def specialize(self,
                   p0: Perception,
                   p1: Perception,
                   leave_specialized=False) -> None:
        for idx in range(len(p1)):
            if leave_specialized:
                if self.effect[idx] != self.cfg.classifier_wildcard:
                    continue

            if p0[idx] != p1[idx]:
                if self.effect[idx] == self.cfg.classifier_wildcard:
                    if p1[idx] != '0.0':
                        self.effect[idx] = '1.0'
                    else:
                        self.effect[idx] = '0.0'

                if p0[idx] != '0.0':
                    self.condition[idx] = '1.0'
                else:
                    self.condition[idx] = '0.0'
