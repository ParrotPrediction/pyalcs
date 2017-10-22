from builtins import isinstance

from alcs.acs2 import Constants as c
from random import sample


class Condition(list):
    """
    Specifies the set of situations (perceptions) in which the classifier
    can be applied.
    """

    def __init__(self, *args):
        if not args:
            list.__init__(self, [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH)
        else:
            list.__init__(self, *args)
            if len(self) != c.CLASSIFIER_LENGTH:
                raise ValueError('Illegal length of condition string')

    def __setitem__(self, idx, value):
        if not isinstance(value, str):
            raise TypeError('Condition should be composed of string objects')

        super(Condition, self).__setitem__(idx, value)

    def __repr__(self):
        return ''.join(map(str, self))

    @property
    def specificity(self) -> int:
        """
        Returns the number of  specific symbols (non-#)

        :return: number of non-general elements
        """
        return sum(1 for comp in self if comp != c.CLASSIFIER_WILDCARD)

    def specialize(self,
                   position: int = None,
                   value: str = None,
                   new_condition=None):

        if position is not None and value is not None:
            self[position] = value

        if new_condition is not None:
            for idx, (oi, ni) in enumerate(zip(self, new_condition)):
                if ni != c.CLASSIFIER_WILDCARD:
                    self[idx] = ni

    def generalize(self, position=None):
        self[position] = c.CLASSIFIER_WILDCARD

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

        for idx, attrib in enumerate(self):
            if attrib != c.CLASSIFIER_WILDCARD \
                    and lst[idx] != c.CLASSIFIER_WILDCARD \
                    and attrib != lst[idx]:
                return False

        return True

    def two_point_crossover(self, other):
        """
        Executes two-point crossover
        :param other: other condition given as list
        """
        left, right = sample(range(0, c.CLASSIFIER_LENGTH), 2)

        if left > right:
            left, right = right, left

        # Extract chromosomes
        chromosome1 = self[left:right]
        chromosome2 = other[left: right]

        # Flip them
        for idx, el in enumerate(range(left, right)):
            self[el] = chromosome2[idx]
            other[el] = chromosome1[idx]

    def equal(self, other):
        return isinstance(other, Condition) \
               and self.specificity == other.specificity \
               and ''.join(self) == ''.join(other)
