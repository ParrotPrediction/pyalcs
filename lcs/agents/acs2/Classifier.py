from __future__ import annotations

from random import randint
from typing import Optional, Union, Callable

from lcs import Perception
from . import Configuration, Condition, Effect, PMark


class Classifier(object):
    def __init__(self,
                 condition: Union[Condition, str, None]=None,
                 action: Optional[int]=None,
                 effect: Union[Effect, str, None]=None,
                 quality: float=0.5,
                 reward: float=0.5,
                 intermediate_reward: float=0.0,
                 numerosity: int=1,
                 experience: int=1,
                 talp=None,
                 tga: int=0,
                 tav: float=0.0,
                 cfg: Optional[Configuration] = None) -> None:

        if cfg is None:
            raise TypeError("Configuration should be passed to Classifier")

        self.cfg = cfg

        def build_perception_string(cls, initial,
                                    length=self.cfg.classifier_length,
                                    wildcard=self.cfg.classifier_wildcard):
            if initial:
                return cls(initial, wildcard=wildcard)

            return cls.empty(wildcard=wildcard, length=length)

        self.condition = build_perception_string(Condition, condition)
        self.action = action
        self.effect = build_perception_string(Effect, effect)

        self.mark = PMark(cfg=self.cfg)

        # Quality - measures the accuracy of the anticipations
        self.q = quality

        # The reward prediction - predicts the reward expected after
        # the execution of action A given condition C
        self.r = reward

        # Intermediate reward
        self.ir = intermediate_reward

        # Numerosity
        self.num = numerosity

        # Experience
        self.exp = experience

        # When ALP learning was triggered
        self.talp = talp

        self.tga = tga

        # Application average
        self.tav = tav

        # I don't know yet what it is
        self.ee = 0

    def q3num(self):
        return pow(self.q, 3) * self.num

    def __eq__(self, other):
        return self.is_similar(other)

    def __hash__(self):
        return hash((str(self.condition), self.action, str(self.effect)))

    def __repr__(self):
        return "{}-{}-{} @ {}".format(self.condition,
                                      self.action,
                                      self.effect,
                                      hex(id(self)))

    @classmethod
    def copy_from(cls, old_cls: Classifier, time: int):
        """
        Copies old classifier with given time (tga, talp).
        Old tav gets replaced with new value.
        New classifier also has no mark.

        Parameters
        ----------
        old_cls: Classifier
            classifier to copy from
        time: int
            time of creation / current epoch

        Returns
        -------
        Classifier
            copied classifier
        """
        new_cls = cls(
            condition=Condition(old_cls.condition, old_cls.cfg),
            action=old_cls.action,
            effect=old_cls.effect,
            quality=old_cls.q,
            reward=old_cls.r,
            intermediate_reward=old_cls.ir,
            cfg=old_cls.cfg)

        new_cls.tga = time
        new_cls.talp = time
        new_cls.tav = old_cls.tav

        return new_cls

    @property
    def fitness(self):
        return self.q * self.r

    @property
    def specified_unchanging_attributes(self) -> int:
        """
        Determines the number of specified unchanging attributes in
        the classifier. An unchanging attribute is one that is anticipated
        not to change in the effect part.

        Returns
        -------
        int
            number of specified unchanging attributes
        """
        spec = 0

        for cpi, epi in zip(self.condition, self.effect):
            if cpi != self.cfg.classifier_wildcard and \
                    epi == self.cfg.classifier_wildcard:
                spec += 1

        return spec

    @property
    def specificity(self):
        return self.condition.specificity / len(self.condition)

    def does_anticipate_change(self) -> bool:
        """
        Checks whether any change in environment is anticipated

        Returns
        -------
        bool
            true if the effect part contains any specified attributes
        """
        return self.effect.number_of_specified_elements > 0

    def is_reliable(self):
        return self.q > self.cfg.theta_r

    def is_inadequate(self):
        return self.q < self.cfg.theta_i

    def update_reward(self, p: float) -> float:
        self.r += self.cfg.beta * (p - self.r)
        return self.r

    def update_intermediate_reward(self, rho) -> float:
        self.ir += self.cfg.beta * (rho - self.ir)
        return self.ir

    def increase_experience(self) -> int:
        self.exp += 1
        return self.exp

    def increase_quality(self) -> float:
        self.q += self.cfg.beta * (1 - self.q)
        return self.q

    def decrease_quality(self) -> float:
        self.q -= self.cfg.beta * self.q
        return self.q

    def specialize(self,
                   previous_situation: Perception,
                   situation: Perception) -> None:
        """
        Specializes the effect part where necessary to correctly anticipate
        the changes from p0 to p1 and returns a condition which specifies
        the attributes which must be specified in the condition part.
        The specific attributes in the returned conditions are set to
        the necessary values.

        Parameters
        ----------
        previous_situation: Perception
        situation: Perception

        Returns
        -------
        """
        for idx, item in enumerate(situation):
            if previous_situation[idx] != situation[idx]:
                self.effect[idx] = situation[idx]
                self.condition[idx] = previous_situation[idx]

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
        if self.condition.does_match(p0):
            if self.action == action:
                if self.does_anticipate_correctly(p0, p1):
                    return True

        return False

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

        Parameters
        ----------
        previous_situation: Perception
            Previous situation
        situation: Perception
            Current situation

        Returns
        -------
        bool
            True if classifier's effect pat anticipates correctly,
            False otherwise
        """
        for idx, item in enumerate(self.effect):
            if item == self.cfg.classifier_wildcard:
                if previous_situation[idx] != situation[idx]:
                    return False
            else:
                if item != situation[idx] \
                        or previous_situation[idx] == situation[idx]:
                    return False

        return True

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
        if self.mark.set_mark_using_condition(self.condition, perception):
            self.ee = 0

    def set_alp_timestamp(self, time: int) -> None:
        """
        Sets the ALP time stamp and the application average parameter.

        Parameters
        ----------
        time: int
            current step
        """
        # TODO p5: write test
        if 1. / self.exp > self.cfg.beta:
            self.tav = (self.tav * self.exp + (time - self.talp)) / (
                self.exp + 1)
        else:
            self.tav += self.cfg.beta * ((time - self.talp) - self.tav)

        self.talp = time

    def is_similar(self, other: Classifier) -> bool:
        """
        Check if classifier is equals to `other` classifier in condition,
        action and effect part.

        Parameters
        ----------
        other: Classifier
            other classifier
        Returns
        -------
        bool
            True if equals, False otherwise
        """
        if self.condition == other.condition and \
                self.action == other.action and \
                self.effect == other.effect:
            return True
        return False

    def is_more_general(self, other: Classifier) -> bool:
        """
        Checks if the classifier is formally more general than `other`.

        Parameters
        ----------
        other: Classifier
            other classifier to compare

        Returns
        -------
        bool
            True if `other` classifier is more general
        """
        if self.condition.specificity < other.condition.specificity:
            return True

        return False

    def generalize_unchanging_condition_attribute(
            self, no_spec: int, randomfunc: Callable=randint) -> bool:
        """
        Generalizes one randomly unchanging attribute in the condition.
        An unchanging attribute is one that is anticipated not to change
        in the effect part.

        Parameters
        ----------
        no_spec: int
            number of unchanging attributes
        randomfunc: Callable
            specifies random function for distinguishing
            which attribute to generalize
        Returns
        -------
        bool
            True if attribute was generalized, False otherwise
        """
        if no_spec == 0:
            return False  # nothing to generalize

        att_idx = randomfunc(0, no_spec - 1)  # id of attribute to generalize
        pos = 0  # current unchanging attribute id

        for idx, (cpi, epi) in enumerate(zip(self.condition, self.effect)):
            if cpi != self.cfg.classifier_wildcard and \
                    epi == self.cfg.classifier_wildcard:

                if att_idx == pos:
                    self.condition.generalize(idx)
                    return True
                else:
                    pos += 1

        return False

    def does_subsume(self, other: Classifier) -> bool:
        """
        Returns if a classifier subsumes `other` classifier

        Parameters
        ----------
        other: Classifier
            other classifier

        Returns
        -------
        bool
            True if `other` classifier is subsumed, False otherwise
        """
        if self._is_subsumer() and \
            self.is_more_general(other) and \
            self.condition.does_match(other.condition) and \
                self.effect == other.effect:
            return True

        return False

    def _is_subsumer(self) -> bool:
        """
        Controls if the classifier satisfies the subsumer criteria.

        Returns
        -------
        bool
            True is classifier can be considered as subsumer,
            False otherwise
        """
        if self.exp > self.cfg.theta_exp:
            if self.is_reliable():
                if not self.is_marked():
                    return True

        return False

    def is_marked(self):
        return self.mark.is_marked()
