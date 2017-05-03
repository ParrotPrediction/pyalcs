from alcs.agent.acs2 import Constants as c


class Condition(list):
    """
    Specifies the set of situations (perceptions) in which the classifier
    can be applied.
    """

    def __init__(self, *args):
        if len(args) > 0:
            list.__init__(self, *args)
        else:
            list.__init__(self, [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH)

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

    def generalize(self, position):
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

        for i, symbol in enumerate(self):
            if symbol != c.CLASSIFIER_WILDCARD and symbol != lst[i]:
                return False

        return True

