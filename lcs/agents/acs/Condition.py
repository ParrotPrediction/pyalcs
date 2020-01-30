from __future__ import annotations

import random
from typing import Callable

from lcs import Perception
from .. import ImmutableSequence


class Condition(ImmutableSequence):
    """
    Specifies the set of situations (perceptions) in which the classifier
    can be applied.
    """

    @property
    def specificity(self) -> int:
        """
        Returns
        -------
        int
            Number of not generic (wildcards) attributes
        """
        return sum(1 for attr in self if attr != self.WILDCARD)

    def specialize_with_condition(self, other: Condition) -> None:
        for idx, new_el in enumerate(other):
            if new_el != self.WILDCARD:
                self[idx] = new_el

    def generalize(self, position=None):
        self[position] = self.WILDCARD

    def generalize_specific_attribute_randomly(
            self, func: Callable = random.choice) -> None:
        """
        Generalizes one randomly selected specified attribute.

        Parameters
        ----------
        func: Callable
            Function for choosing which ID to generalize from the list of
            available ones
        """
        specific_ids = [ci for ci, c in enumerate(self) if c != self.WILDCARD]

        if len(specific_ids) > 0:
            ridx = func(specific_ids)
            self.generalize(ridx)

    def does_match(self, p: Perception) -> bool:
        """
        Check if condition match other list such as perception or another
        condition.

        Parameters
        ----------
        p: Union[Perception, Condition]
            perception or condition object

        Returns
        -------
        bool
            True if condition match given list, False otherwise
        """
        for ci, oi in zip(self, p):
            if ci != self.WILDCARD and oi != self.WILDCARD and ci != oi:
                return False

        return True

    def subsumes(self, other: Condition) -> bool:
        for ci, oi in zip(self, other):
            if ci != self.WILDCARD and oi != self.WILDCARD and ci != oi:
                return False

        return True

    def get_backwards_anticipation(self, perception: Perception) -> Perception:
        """
        Returns the believed backwards anticipation. Hereby, the condition
        is treated like an effect part.
        :param perception:
        :return:
        """
        ant = list(perception)

        for idx, item in enumerate(self):
            if item != self.WILDCARD:
                ant[idx] = item

        return Perception(ant)
