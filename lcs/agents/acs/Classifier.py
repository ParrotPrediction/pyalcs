from __future__ import annotations

import logging
from typing import Optional, Union, List

from lcs import Perception
from lcs.agents.acs import Condition, Configuration, PMark, Effect

logger = logging.getLogger(__name__)


class Classifier:
    __slots__ = ['condition', 'action', 'effect', 'mark', 'q', 'r',
                 'talp', 'tav', 'cfg']

    def __init__(self,
                 condition: Union[Condition, str, None] = None,
                 action: Optional[int] = None,
                 effect: Union[Effect, str, None] = None,
                 quality: float = 0.5,  # predicts the accuracy of anticipation
                 reward: float = 0.5,
                 talp=None,
                 tav: float = 0.0,
                 cfg: Optional[Configuration] = None) -> None:

        if cfg is None:
            raise TypeError("Configuration should be passed to Classifier")

        self.cfg = cfg

        def build_perception_string(cls, initial,
                                    length=self.cfg.classifier_length):
            if initial:
                return cls(initial)

            return cls.empty(length=length)

        self.condition = build_perception_string(Condition, condition)
        self.action = action
        self.effect = build_perception_string(Effect, effect)

        self.mark = PMark(cfg=self.cfg)

        # Quality - measures the accuracy of the anticipations
        self.q = quality

        # The reward prediction - predicts the reward expected after
        # the execution of action A given condition C
        self.r = reward

        self.talp = talp  # When ALP learning was triggered
        self.tav = tav  # Application average

    def __eq__(self, other):
        if self.condition == other.condition and \
            self.action == other.action and \
                self.effect == other.effect:
            return True

        return False

    def __hash__(self):
        return hash((str(self.condition), self.action, str(self.effect)))

    def __repr__(self):
        return f"{self.condition} " \
               f"{self.action} " \
               f"{self.effect} " \
               f"{'(' + str(self.mark) + ')':21} q: {self.q:<5.3} " \
               f"r: {self.r:<6.4} f: {self.fitness:<6.4}"

    @classmethod
    def general(cls, action: int, cfg):
        return cls(condition=None, action=action, effect=None, cfg=cfg)

    @classmethod
    def build_corrected(cls,
                        old: Classifier,
                        p0: Perception,
                        p1: Perception) -> Classifier:
        """
        Constructs the classifier for "correctable case".
        C_new and E_new will be different from the old classifier in the
        non-matching components of p0 and p1.

        There
        - E_new will be equal to p1
        - C_new will be equal to p0
        respectively.

        Parameters
        ----------
        old: Classifier
            Old classifier that will be cloned and changed
        p0: Perception
            previous perception
        p1: Perception
            perception

        Returns
        -------
        Classifier
            new corrected classifier

        """
        assert p0 != p1
        new_c = Condition(old.condition)
        new_e = Effect(old.effect)

        for idx, (ci, ei, p0i, p1i) in \
                enumerate(zip(old.condition, old.effect, p0, p1)):

            if p0i != p1i:
                new_c[idx] = p0i
                new_e[idx] = p1i

        return Classifier(
            condition=new_c,
            action=old.action,
            effect=new_e,
            cfg=old.cfg
        )

    @property
    def fitness(self):
        if self.cfg.fitness_fcn:
            return self.cfg.fitness_fcn(self)

        return self.q * self.r

    @property
    def specified_unchanging_attributes(self) -> List[int]:
        """
        Determines the specified unchanging attributes in the classifier.
        An unchanging attribute is one that is anticipated not to change
        in the effect part.

        Returns
        -------
        List[int]
            list of specified unchanging attributes indices
        """
        indices = []

        for idx, (cpi, epi) in enumerate(zip(self.condition, self.effect)):
            if cpi != self.cfg.classifier_wildcard and \
                    epi == self.cfg.classifier_wildcard:
                indices.append(idx)

        return indices

    @property
    def specificity(self):
        return self.condition.specificity / len(self.condition)

    def is_general(self):
        cl_length = self.cfg.classifier_length

        return self.condition == Condition.empty(cl_length) \
            and self.effect == Effect.empty(cl_length)

    def does_anticipate_change(self) -> bool:
        """
        Checks whether any change in environment is anticipated

        Returns
        -------
        bool
            true if the effect part contains any specified attributes
        """
        return self.effect.specify_change

    def can_be_corrected(self, p0: Perception, p1: Perception) -> bool:
        """
        If all components of C and E for which p0 != p1 are wildcards then
        classifier can be corrected.

        Parameters
        ----------
        p0: Perception
            previous perception
        p1: Perception
            perception

        Returns
        -------
        bool
            True if classifier can be corrected, False otherwise
        """
        assert p0 != p1  # change must be present by definition

        wildcard = self.cfg.classifier_wildcard
        for ci, ei, p0i, p1i in zip(self.condition, self.effect, p0, p1):
            if p0i != p1i:
                if ci != wildcard or ei != wildcard:
                    # Exit the function if negative condition is found
                    return False

        return True

    def is_reliable(self) -> bool:
        return self.q > self.cfg.theta_r

    def is_inadequate(self) -> bool:
        return self.q < self.cfg.theta_i

    def decrease_quality(self):
        self.q *= (1 - self.cfg.beta)

    def increase_quality(self):
        self.q = (1 - self.cfg.beta) * self.q + self.cfg.beta

    def specialize(self,
                   p0: Perception,
                   p1: Perception,
                   leave_specialized=False) -> None:
        """
        Specializes the effect part where necessary to correctly anticipate
        the changes from p0 to p1.

        Parameters
        ----------
        p0: Perception
            previous_situation
        p1: Perception
            situation
        leave_specialized: bool
            Requires the effect attribute to be a wildcard to specialize it.
            By default false
        """
        for idx in range(len(p1)):
            if leave_specialized:
                if self.effect[idx] != self.cfg.classifier_wildcard:
                    # If we have a specialized attribute don't change it.
                    continue

            if p0[idx] != p1[idx]:
                if self.effect[idx] == self.cfg.classifier_wildcard:
                    self.effect[idx] = p1[idx]

                self.condition[idx] = p0[idx]

    def predicts_successfully(self,
                              p0: Perception,
                              action: int,
                              p1: Perception) -> bool:
        """
        Check if classifier matches previous situation `p0`,
        has action `action` and predicts the effect `p1`

        Parameters
        ----------
        p0: Perception
            previous situation
        action: int
            action
        p1: Perception
            anticipated situation after execution action

        Returns
        -------
        bool
            True if classifier makes successful predictions, False otherwise
        """
        if self.does_match(p0):
            if self.action == action:
                if self.does_anticipate_correctly(p0, p1):
                    return True

        return False

    def does_anticipate_correctly(self,
                                  p0: Perception,
                                  p1: Perception) -> bool:
        """
        Checks anticipation. While the pass-through symbols in the effect part
        of a classifier directly anticipate that these attributes stay the same
        after the execution of an action, the specified attributes anticipate
        a change to the specified value. Thus, if the perceived value did not
        change to the anticipated but actually stayed at the value, the
        classifier anticipates incorrectly.

        Parameters
        ----------
        p0: Perception
            Previous situation
        p1: Perception
            Current situation

        Returns
        -------
        bool
            True if classifier's effect pat anticipates correctly,
            False otherwise
        """

        return self.effect.anticipates_correctly(p0, p1)

    def set_mark(self, perception: Perception) -> None:
        """
        Marks classifier with given perception taking into consideration its
        condition.

        Specializes the mark in all attributes which are not specified
        in the conditions, yet

        Parameters
        ----------
        perception: Perception
            current situation
        """
        self.mark.set_mark_using_condition(self.condition, perception)

    def is_more_general(self, other: Classifier) -> bool:
        """
        Checks if the classifiers condition is formally
        more general than `other`s.

        Parameters
        ----------
        other: Classifier
            other classifier to compare

        Returns
        -------
        bool
            True if `other` classifier is more general
        """
        return self.condition.specificity < other.condition.specificity

    def is_marked(self):
        return self.mark.is_marked()

    def does_match(self, situation: Perception) -> bool:
        """
        Returns if the classifier matches the situation.
        :param situation:
        :return:
        """
        return self.condition.does_match(situation)
