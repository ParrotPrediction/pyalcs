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

    def get_best_anticipation(self, perception):
        """
        Returns the most probable anticipation of the effect part.
        This is usually the normal anticipation. However, if PEEs are activated, the most probable
        value of each attribute is taken as the anticipation.
        :param perception: Perception
        :return:
        """
        # TODO: implement the rest after PEEs are implemented ('getBestChar' function)
        ant = Perception(perception)
        for idx, item in enumerate(self):
            if item != self.cfg.classifier_wildcard:
                ant[idx] = item
        return ant

    def does_specify_only_changes_backwards(self, back_anticipation, situation):
        """
        Returns if the effect part specifies at least one of the percepts.
        An PEE attribute never specifies the corresponding percept.
        :param back_anticipation: Perception
        :param situation: Perception
        :return:
        """
        for idx, (item, back_ant, sit) in enumerate(zip(self, back_anticipation, situation)):
            if item == self.cfg.classifier_wildcard and back_ant != sit:
                # change anticipated backwards although no change should occur
                return False
            # TODO: if PEEs are implemented, 'isEnhanced()' should be added to the condition below
            if item != self.cfg.classifier_wildcard and item == back_ant:
                return False
        return True

    def does_match(self, perception, other_perception):
        """
        Returns if the effect matches the perception.
        Hereby, the specified attributes are compared with perception.
        Where the effect part has got #-symbols perception and other_perception are compared.
        If they are not equal the effect part does not match.
        :param perception: Perception
        :param other_perception: Perception
        :return:
        """
        for (item, percept, percept2) in zip(self, perception, other_perception):
            if item == self.cfg.classifier_wildcard and percept != percept2:
                return False
            elif item != self.cfg.classifier_wildcard and item != percept:
                return False
        return True
