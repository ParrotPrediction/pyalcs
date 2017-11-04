from alcs import Perception
from alcs.acs2.AbstractCondition import AbstractCondition


class Effect(AbstractCondition):
    """
    Anticipates the effects that the classifier 'believes'
    to be caused by the specified action.
    """

    @property
    def number_of_specified_elements(self) -> int:
        """
        :return: number of specified components
        """
        return sum(1 for comp in self if comp != self.cfg.classifier_wildcard)

    def does_anticipate_correctly(self,
                                  previous_situation: Perception,
                                  situation: Perception) -> bool:
        """
        Checks anticipation. While the pass-through symbols in the effect part
        of a classifier directly anticipate that these attributes stay the same
        after the execution of an action, the specified attributes anticipate
        a change to the specified value. Thus, if the perceived value did not
        change to the anticipated but actually stayed at the value, the
        classifier anticipates incorrectly.

        :param previous_situation:
        :param situation:
        :return: True if classifier anticipates correctly, False otherwise
        """
        # TODO p1: write some tests
        for idx, item in enumerate(self):
            if item == self.cfg.classifier_wildcard:
                if previous_situation[idx] != situation[idx]:
                    return False
            else:
                if (item != situation[idx] or
                        previous_situation[idx] == situation[idx]):
                    return False

        return True

    def is_specializable(self, p0: Perception, p1: Perception) -> bool:
        """
        Determines if the effect part can be modified to anticipate
        changes from `p0` to `p1` correctly by only specializing attributes.

        :param p0: previous perception
        :param p1: current perception
        :return: True if specializable, false otherwise
        """
        for p0i, p1i, ei in zip(p0, p1, self):
            if ei != self.cfg.classifier_wildcard:
                if ei != p1i or p0i == p1i:
                    return False

        return True
