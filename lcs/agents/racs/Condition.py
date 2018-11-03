from __future__ import annotations

import random
import statistics
from copy import copy
from typing import Callable

from lcs import Perception
from lcs.representations.visualization import visualize
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

    @property
    def cover_ratio(self) -> float:
        """
        Calculates the perception space covered by condition attribute.
        An arithmetic average is taken over all condition attributes

        Returns
        -------
        float
            A value between [0.0, 1.0], where
            0.0 means that condition is extremely narrow
            1.0 means that condition is maximally general
        """
        maximum_span = self.cfg.encoder.range[1]
        return statistics.mean(r.bound_span / maximum_span for r in self)

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

    def generalize(self, idx: int) -> None:
        """
        Broadens the range of specific condition attribute
        to the maximum range.

        Parameters
        ----------
        idx: int
            Id of the attribute to generalize
        """
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

    def subsumes(self, other: Condition):
        return all(ci.incorporates(oi) for ci, oi in zip(self, other))

    def __repr__(self):
        return "|".join(visualize(
            (ubr.lower_bound, ubr.upper_bound),
            self.cfg.encoder.range) for ubr in self)
