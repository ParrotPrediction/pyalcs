from .. import PerceptionString


class Condition(PerceptionString):
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
        return sum(1 for comp in self if comp != self.wildcard)

    def specialize(self,
                   position: int = None,
                   value: str = None,
                   new_condition=None):

        if position is not None and value is not None:
            self[position] = value

        if new_condition is not None:
            for idx, (oi, ni) in enumerate(zip(self, new_condition)):
                if ni != self.wildcard:
                    self[idx] = ni

    def generalize(self, position=None):
        self[position] = self.wildcard

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
