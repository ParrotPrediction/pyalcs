from alcs import Perception
from alcs.acs2 import AbstractCondition
from random import sample


class Condition(AbstractCondition):
    """
    Specifies the set of situations (perceptions) in which the classifier
    can be applied.
    """

    @property
    def specificity(self) -> int:
        """
        Returns the number of  specific symbols (non-#)

        :return: number of non-general elements
        """
        return sum(1 for comp in self if comp != self.cfg.classifier_wildcard)

    def specialize(self,
                   position: int = None,
                   value: str = None,
                   new_condition=None):

        if position is not None and value is not None:
            self[position] = value

        if new_condition is not None:
            for idx, (oi, ni) in enumerate(zip(self, new_condition)):
                if ni != self.cfg.classifier_wildcard:
                    self[idx] = ni

    def generalize(self, position=None):
        self[position] = self.cfg.classifier_wildcard

    def does_match(self, lst: list) -> bool:
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
            if attrib != self.cfg.classifier_wildcard \
                    and lst[idx] != self.cfg.classifier_wildcard \
                    and attrib != lst[idx]:
                return False

        return True

    def two_point_crossover(self, other, samplefunc=sample):
        """
        Executes two-point crossover
        :param samplefunc:
        :param other: other condition given as list
        """
        left, right = samplefunc(range(0, self.cfg.classifier_length + 1), 2)

        if left > right:
            left, right = right, left

        # Extract chromosomes
        chromosome1 = self[left:right]
        chromosome2 = other[left:right]

        # Flip them
        for idx, el in enumerate(range(left, right)):
            self[el] = chromosome2[idx]
            other[el] = chromosome1[idx]
