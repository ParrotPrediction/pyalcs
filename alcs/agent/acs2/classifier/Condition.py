from alcs.agent.acs2 import Constants as c


class Condition(object):
    """
    Specifies the set of situations (perceptions) in which the classifier
    can be applied.
    """

    def __init__(self):
        self.list = [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH

    def __repr__(self):
        return ''.join(map(str, self.list))

    def __getitem__(self, item):
        return self.list[item]

    def specialize(self, position: int, value: int):
        self.list[position] = value

    def generalize(self, position):
        self.list[position] = c.CLASSIFIER_WILDCARD

    def does_match(self, perception: list) -> bool:
        """
        Check if condition match given perception

        :param perception: perception given as list
        :return: True if condition match given perception, false otherwise
        """
        if len(perception) != len(self.list):
            raise ValueError('Perception and classifier condition '
                             'length is different')

        for i, symbol in enumerate(perception):
            if symbol != c.CLASSIFIER_WILDCARD and symbol != perception[i]:
                return False

        return True

    def number_of_specified_elements(self) -> int:
        """
        Returns the number of  specific symbols (non-#)

        :return: number of non-general elements
        """
        return sum(1 for e in self.list if e != c.CLASSIFIER_WILDCARD)

    def get_specificity(self) -> float:
        """
        Return information what percentage of condition elements are
        specific.

        :return: number from [0,1]
        """
        return self.number_of_specified_elements() / c.CLASSIFIER_LENGTH

