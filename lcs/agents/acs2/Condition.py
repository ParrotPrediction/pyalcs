from __future__ import annotations

import random
from typing import Callable, Union

from lcs import Perception
from .. import PerceptionString


class Condition(PerceptionString):
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
        return sum(1 for comp in self if comp != self.wildcard)

    def specialize_with_condition(self, other: Condition) -> None:
        for idx, new_el in enumerate(other):
            if new_el != self.wildcard:
                self[idx] = new_el

    def generalize(self, position=None):
        self[position] = self.wildcard

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
        specific_ids = [ci for ci, c in enumerate(self) if c != self.wildcard]

        if len(specific_ids) > 0:
            ridx = func(specific_ids)
            self.generalize(ridx)

    def does_match(self, other: Union[Perception, Condition]) -> bool:
        """
        Check if condition match other list such as perception or another
        condition.

        Parameters
        ----------
        other: Union[Perception, Condition]
            perception or condition object

        Returns
        -------
        bool
            True if condition match given list, False otherwise
        """
        for ci, oi in zip(self, other):
            if ci != self.wildcard and oi != self.wildcard and ci != oi:
                return False

        return True

    def get_backwards_anticipation(self, perception):
        """
        Returns the believed backwards anticipation. Hereby, the condition
        is treated like an effect part.
        :param perception:
        :return:
        """
        ant = list(perception)
        for idx, item in enumerate(self):
            if item != self.wildcard:
                ant[idx] = item
        return Perception(ant)
