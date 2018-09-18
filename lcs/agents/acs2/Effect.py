from lcs import Perception
from .. import PerceptionString


class Effect(PerceptionString):
    """
    Anticipates the effects that the classifier 'believes'
    to be caused by the specified action.
    """

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
        return any(True for e in self if e != self.wildcard)

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
            if ei != self.wildcard:
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
