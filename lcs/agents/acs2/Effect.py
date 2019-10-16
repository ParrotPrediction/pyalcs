from __future__ import annotations

from lcs import Perception
from lcs.agents.acs2 import Configuration
from lcs.agents.acs2 import ProbabilityEnhancedAttribute
from .. import PerceptionString


DETAILED_PEE_PRINTING = True


class Effect(PerceptionString):
    """
    Anticipates the effects that the classifier 'believes'
    to be caused by the specified action.
    """

    def __init__(self, observation, wildcard='#', oktypes=(str, dict)):
        # Convert dict to ProbabilityEnhancedAttribute
        if not all(isinstance(attr, ProbabilityEnhancedAttribute)
                   for attr in observation):
            observation = (ProbabilityEnhancedAttribute(attr)
                           if isinstance(attr, dict)
                           else attr
                           for attr in observation)

        super().__init__(observation, wildcard, oktypes)

    @classmethod
    def enhanced_effect(cls, effect1, effect2,
                        q1: float = 0.5, q2: float = 0.5,
                        perception: PerceptionString = None,
                        wildcard='#'):
        """
        Create a new enhanced effect part.
        """
        result = cls(observation=effect1, wildcard=wildcard)
        for i, attr2 in enumerate(effect2):
            attr1 = effect1[i]
            if attr1 == wildcard and attr2 == wildcard:
                continue
            if attr1 == wildcard:
                attr1 = perception[i]
            if attr2 == wildcard:
                attr2 = perception[i]

            result[i] = ProbabilityEnhancedAttribute.merged_attributes(
                attr1, attr2, q1, q2)
        return result

    @classmethod
    def for_perception_change(cls,
                              p0: PerceptionString,
                              p1: PerceptionString,
                              cfg: Configuration):
        """
        Create an Effect that represents the change from perception p0
        to perception p1.
        """

        # Start with the resulting perception
        result = cls(observation=p1, wildcard=cfg.classifier_wildcard)

        # Insert wildcard characters where necessary
        for idx, eitem in enumerate(result):
            if p0[idx] == p1[idx]:
                result[idx] = cfg.classifier_wildcard

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
        if self.is_enhanced():
            return True
        else:
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
        if self.is_enhanced():
            return True

        for p0i, p1i, ei in zip(p0, p1, self):
            if ei != self.wildcard:
                if ei != p1i or p0i == p1i:
                    return False

        return True

    def get_best_anticipation(self, perception: Perception) -> Perception:
        """
        Returns the most probable anticipation of the effect part.
        This is usually the normal anticipation. However, if PEEs are
        activated, the most probable value of each attribute is
        taken as the anticipation.
        :param perception: Perception
        :return:
        """
        # TODO: implement the rest after PEEs are implemented
        # ('getBestChar' function)
        ant = list(perception)
        for idx, item in enumerate(self):
            if item != self.wildcard:
                ant[idx] = item
        return Perception(ant)

    def does_specify_only_changes_backwards(self,
                                            back_anticipation: Perception,
                                            situation: Perception) -> bool:
        """
        Returns if the effect part specifies at least one of the percepts.
        An PEE attribute never specifies the corresponding percept.
        :param back_anticipation: Perception
        :param situation: Perception
        :return:
        """
        for item, back_ant, sit in zip(self, back_anticipation, situation):
            if item == self.wildcard and back_ant != sit:
                # change anticipated backwards although no change should occur
                return False
            # TODO: if PEEs are implemented, 'isEnhanced()' should be added
            # to the condition below
            if item != self.wildcard and item == back_ant:
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
            if item == self.wildcard and percept != percept2:
                return False
            elif item != self.wildcard and item != percept:
                return False
        return True

    def subsumes(self, other: Effect) -> bool:
        return self == other

    def is_enhanced(self) -> bool:
        """
        Checks whether any element of the Effect is Probability-Enhanced.
        str elements of the Effect are not Enhanced,
        ProbabilityEnhancedAttribute elements are Enhanced.

        :return: True if this is a Probability-Enhanced Effect, False otherwise
        """
        # Sanity check
        assert not any(isinstance(elem, dict) and
                       not isinstance(elem, ProbabilityEnhancedAttribute)
                       for elem in self)

        return any(isinstance(elem, ProbabilityEnhancedAttribute)
                   for elem in self)

    def reduced_to_non_enhanced(self):
        if not self.is_enhanced():
            return self

        result = Effect(self)
        for i, elem in enumerate(result):
            if isinstance(elem, ProbabilityEnhancedAttribute):
                result[i] = self[i].get_best_symbol()
        return result

    def update_enhanced_effect_probs(self, perception: Perception,
                                     update_rate: float):
        for i, elem in enumerate(self):
            if isinstance(elem, ProbabilityEnhancedAttribute):
                elem.make_compact()
                effect_symbol = perception[i]
                elem.increase_probability(effect_symbol, update_rate)

    def __str__(self):
        if DETAILED_PEE_PRINTING:
            return ''.join(str(attr) for attr in self)
        else:
            if self.is_enhanced():
                return '(PEE)' + ''.join(attr for attr
                                         in self.reduced_to_non_enhanced())
            else:
                assert all(isinstance(attr, str) for attr in self)
                return ''.join(attr for attr in self)
