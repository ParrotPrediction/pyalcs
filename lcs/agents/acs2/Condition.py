import random
from typing import Callable

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

    def specialize(self,
                   position: int=None,
                   value: str=None,
                   new_condition=None):

        if position is not None and value is not None:
            self[position] = value

        if new_condition is not None:
            for idx, (oi, ni) in enumerate(zip(self, new_condition)):
                if ni != self.wildcard:
                    self[idx] = ni

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

    def does_match(self, lst) -> bool:
        """
        Check if condition match other list such as perception or another
        condition.

        :param lst: perception or condition given as list
        :return: True if condition match given list, false otherwise
        """
        if len(self) != len(lst):
            raise ValueError('Cannot execute `does_match` '
                             'because lengths are different')

        # TODO zip can be used instead
        for idx, attrib in enumerate(self):
            if attrib != self.wildcard \
                    and lst[idx] != self.wildcard \
                    and attrib != lst[idx]:
                return False

        return True
