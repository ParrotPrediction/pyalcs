from alcs.agent import Perception
from alcs.agent.acs2 import Constants as c


class Effect(list):
    """
    Anticipates the effects that the classifier 'believes'
    to be caused by the specified action.
    """
    def __init__(self, *args):
        if len(args) > 0:
            list.__init__(self, *args)
        else:
            list.__init__(self, [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH)

    def __setitem__(self, idx, value):
        if not isinstance(value, str):
            raise TypeError('Effect should be composed of string objects')

        super(Effect, self).__setitem__(idx, value)

    def __repr__(self):
        return ''.join(map(str, self))

    @property
    def number_of_specified_elements(self) -> int:
        """
        :return: number of specific components
        """
        return sum(1 for comp in self if comp != c.CLASSIFIER_WILDCARD)

    def is_specializable(self, p0: Perception, p1: Perception) -> bool:
        """
        Determines if the effect part is specializable.

        :param p0: previous perception
        :param p1: current perception
        :return: True if specializable, false otherwise
        """
        for p0i, p1i, ei in zip(p0, p1, self):
            if ei != c.CLASSIFIER_WILDCARD:
                if ei != p1i or p0i == p1i:
                    return False

        return True

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
        for idx, item in enumerate(self):
            if item == c.CLASSIFIER_WILDCARD:
                if previous_situation[idx] != situation[idx]:
                    return False
            else:
                if (item != situation[idx] or
                        previous_situation[idx] == situation[idx]):
                    return False

        return True
