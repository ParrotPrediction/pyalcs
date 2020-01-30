from lcs import Perception
from lcs.agents import ImmutableSequence


class Effect(ImmutableSequence):
    """
    Anticipates the effects that the classifier 'believes'
    to be caused by the specified action.
    """

    def __init__(self, observation):
        super().__init__(observation)

    def subsumes(self, other) -> bool:
        return self == other

    @property
    def specify_change(self) -> bool:
        """
        Checks whether there is any attribute in the effect part that
        is not "pass-through" - so predicts a change.

        Returns
        -------
        bool
            True if the effect part predicts a change, False otherwise
        """
        return any(True for e in self if e != self.WILDCARD)

    @classmethod
    def item_anticipate_change(cls, item, p0_item, p1_item) -> bool:
        if item == cls.WILDCARD:
            if p0_item != p1_item:
                return False
        else:
            if p0_item == p1_item:
                return False

            if item != p1_item:
                return False

        # All checks passed
        return True

    def is_specializable(self, p0: Perception, p1: Perception) -> bool:
        """
        Determines if the effect part can be modified to anticipate
        changes from `p0` to `p1` correctly by only specializing attributes.

        Parameters
        ----------
        p0: Perception
            previous perception
        p1: Perception
            current perception

        Returns
        -------
        bool
            True if specializable, false otherwise
        """
        for p0i, p1i, ei in zip(p0, p1, self):
            if ei != self.WILDCARD:
                if ei != p1i or p0i == p1i:
                    return False

        return True

    def does_match(self,
                   perception: Perception,
                   other_perception: Perception) -> bool:
        """
        Returns if the effect matches the perception.
        Hereby, the specified attributes are compared with perception.
        Where the effect part has got #-symbols perception and other_perception
        are compared. If they are not equal the effect part does not match.
        :param perception: Perception
        :param other_perception: Perception
        :return:
        """
        for (item, percept, percept2) in zip(self, perception,
                                             other_perception):
            if item == self.WILDCARD and percept != percept2:
                return False
            elif item != self.WILDCARD and item != percept:
                return False

        return True

    def anticipates_correctly(self, p0: Perception, p1: Perception) -> bool:
        return all(self.item_anticipate_change(eitem, p0[idx], p1[idx])
                   for idx, eitem in enumerate(self))

    def __str__(self):
        assert all(isinstance(attr, str) for attr in self)
        return ''.join(attr for attr in self)
