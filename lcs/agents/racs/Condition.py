from __future__ import annotations

import random
from copy import copy
from typing import Callable

from lcs import Perception
from . import Configuration
from .. import PerceptionString


class Condition(PerceptionString):

    def __init__(self, lst, cfg: Configuration) -> None:
        self.cfg = cfg
        super().__init__(lst, cfg.classifier_wildcard, cfg.oktypes)

    @classmethod
    def generic(cls, cfg: Configuration):
        ps_str = [copy(cfg.classifier_wildcard) for _
                  in range(cfg.classifier_length)]
        return cls(ps_str, cfg)

    @property
    def specificity(self) -> int:
        """
        Returns
        -------
        int
            Number of not generic (wildcards) attributes
        """
        return sum(1 for c in self if c != self.wildcard)

    def specialize_with_condition(self, other: Condition) -> None:
        """
        Specializes the existing condition with specified attributes from
        new condition string.

        Parameters
        ----------
        other: Condition
            New condition string. Most probably the diff from ALP
        """
        for idx, new_el in enumerate(other):
            if new_el != self.wildcard:
                self[idx] = new_el

    def generalize(self, idx: int):
        self[idx] = self.cfg.classifier_wildcard

    def generalize_specific_attribute_randomly(
            self, func: Callable=random.choice) -> None:
        """
        Generalizes one randomly selected specified attribute.

        Parameters
        ----------
        func: Callable
            Function for choosing which ID to generalize from the list of
            available ones
        """
        specific_ids = [ci for ci, c in enumerate(self) if c != self.wildcard]

        if len(specific_ids) > 0:
            ridx = func(specific_ids)
            self.generalize(ridx)

    def does_match(self, perception: Perception):
        encoded_perception = map(self.cfg.encoder.encode, perception)
        return all(p in ubr for p, ubr in zip(encoded_perception, self))
