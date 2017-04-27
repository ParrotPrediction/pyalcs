from alcs.agent.acs3 import Constants as c


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
    def number_of_specified_elements(self) -> int:
        """
        Returns the number of  specific symbols (non-#)

        :return: number of non-general elements
        """
        return sum(1 for comp in self if comp != c.CLASSIFIER_WILDCARD)

    @property
    def specificity(self) -> float:
        """
        Return information what percentage of condition elements are
        specific.

        :return: number from [0,1]
        """
        return self.number_of_specified_elements / len(self)

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

    def does_match(self, perception: list) -> bool:
        """
        Check if condition match given perception

        :param perception: perception given as list
        :return: True if condition match given perception, false otherwise
        """
        if len(perception) != len(self):
            raise ValueError('Perception and condition length is different')

        for i, symbol in enumerate(self):
            if symbol != c.CLASSIFIER_WILDCARD and symbol != perception[i]:
                return False

        return True

