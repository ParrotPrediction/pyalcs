from __future__ import annotations

import logging
import random
from typing import Optional, Union, Callable, List

from lcs import Perception
from . import Configuration, Condition, Effect, PMark, matching
from . import ProbabilityEnhancedAttribute


logger = logging.getLogger(__name__)


class Classifier:

    __slots__ = ['condition', 'action', 'effect', 'mark', 'q', 'r',
                 'ir', 'num', 'exp', 'talp', 'tga', 'tav', 'ee', 'cfg']

    def __init__(self,
                 condition: Union[Condition, str, None]=None,
                 action: Optional[int]=None,
                 effect: Union[Effect, str, None]=None,
                 quality: float=0.5,
                 reward: float=0.5,
                 immediate_reward: float=0.0,
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

        # Immediate reward
        self.ir = immediate_reward

        # Numerosity
        self.num = numerosity

        # Experience
        self.exp = experience

        # When ALP learning was triggered
        self.talp = talp

        self.tga = tga

        # Application average
        self.tav = tav

        # Whether classifier is enhanceable
        self.ee = False

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
               f"{str(self.effect):16} " \
               f"{'(' + str(self.mark) + ')':21} q: {self.q:<5.3} " \
               f"r: {self.r:<6.4} ir: {self.ir:<6.4} f: {self.fitness:<6.4} " \
               f"exp: {self.exp:<3} tga: {self.tga:<5} talp: {self.talp:<5} " \
               f"tav: {self.tav:<6.3} num: {self.num}"

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
            condition=Condition(old_cls.condition),
            action=old_cls.action,
            effect=old_cls.effect,
            quality=old_cls.q,
            reward=old_cls.r,
            immediate_reward=old_cls.ir,
            cfg=old_cls.cfg)

        new_cls.tga = time
        new_cls.talp = time
        new_cls.tav = old_cls.tav

        return new_cls

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
            if isinstance(epi, ProbabilityEnhancedAttribute):
                if cpi != self.cfg.classifier_wildcard and \
                        epi.does_contain(cpi):
                    indices.append(idx)
            else:
                if cpi != self.cfg.classifier_wildcard and \
                        epi == self.cfg.classifier_wildcard:
                    indices.append(idx)

        return indices

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
        return self.effect.specify_change

    def is_reliable(self) -> bool:
        return self.q > self.cfg.theta_r

    def is_inadequate(self) -> bool:
        return self.q < self.cfg.theta_i

    def is_enhanceable(self):
        return self.ee

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
                   situation: Perception,
                   leave_specialized=False) -> None:
        """
        Specializes the effect part where necessary to correctly anticipate
        the changes from p0 to p1.

        Parameters
        ----------
        previous_situation: Perception
        situation: Perception
        leave_specialized: bool
            Requires the effect attribute to be a wildcard to specialize it.
            By default false
        """
        for idx in range(len(situation)):
            if leave_specialized:
                if self.effect[idx] != self.cfg.classifier_wildcard:
                    # If we have a specialized attribute don't change it.
                    continue

            if previous_situation[idx] != situation[idx]:
                if self.effect[idx] == self.cfg.classifier_wildcard:
                    self.effect[idx] = situation[idx]
                else:
                    if not isinstance(self.effect[idx],
                                      ProbabilityEnhancedAttribute):
                        self.effect[idx] = ProbabilityEnhancedAttribute(
                            self.effect[idx])
                    self.effect[idx].insert_symbol(situation[idx])

                self.condition[idx] = previous_situation[idx]

    def merge_with(self, other_classifier, perception, time):
        assert self.cfg.do_pee

        result = Classifier(cfg=self.cfg)

        result.condition = Condition(self.condition)
        result.condition.specialize_with_condition(other_classifier.condition)

        # action is an int, so we can assign directly
        result.action = self.action

        result.effect = Effect.enhanced_effect(
            self.effect, other_classifier.effect,
            self.q, other_classifier.q,
            perception)

        result.mark = PMark(cfg=self.cfg)

        result.r = (self.r + other_classifier.r) / 2.0
        result.q = (self.q + other_classifier.q) / 2.0

        # This 0.5 is Q_INI constant in the original C++ code
        if result.q < 0.5:
            result.q = 0.5

        result.num = 1
        result.tga = time
        result.talp = time
        result.tav = 0
        result.exp = 1

        result.ee = False

        return result

    def reverse_increase_quality(self):
        self.q = (self.q - self.cfg.beta) / (1.0 - self.cfg.beta)

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
        def effect_item_is_correct(effect_item, p0_item, p1_item):
            if not isinstance(effect_item, ProbabilityEnhancedAttribute):
                if effect_item == self.cfg.classifier_wildcard:
                    if p0_item != p1_item:
                        return False
                else:
                    if p0_item == p1_item:
                        return False

                    if effect_item != p1_item:
                        return False
            else:
                if not effect_item.does_contain(p1_item):
                    return False

            # All checks passed
            return True

        return all(effect_item_is_correct(
            eitem, previous_situation[idx], situation[idx])
            for idx, eitem in enumerate(self.effect))

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
            self.ee = False

    def update_application_average(self, time: int) -> None:
        """
        Sets the ALP time stamp and the application average parameter.
        Uses the Moyenne Adaptive Modifee (MAM) technique.

        Parameters
        ----------
        time: int
            current time step
        """
        # TODO p5: write test
        last_applied = time - self.talp - self.tav

        if self.exp < 1. / self.cfg.beta:
            # average of values seen so far
            self.tav += last_applied / self.exp
        else:
            self.tav += self.cfg.beta * last_applied

        self.talp = time

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

    def generalize_unchanging_condition_attribute(
            self, randomfunc: Callable=random.choice) -> bool:
        """
        Generalizes one randomly unchanging attribute in the condition.
        An unchanging attribute is one that is anticipated not to change
        in the effect part.

        Parameters
        ----------
        randomfunc: Callable
            function returning attribute index to generalize
        Returns
        -------
        bool
            True if attribute was generalized, False otherwise
        """
        if len(self.specified_unchanging_attributes) > 0:
            ridx = randomfunc(self.specified_unchanging_attributes)
            self.condition.generalize(ridx)
            return True

        return False

    def is_marked(self):
        return self.mark.is_marked()

    def does_match(self, situation: Perception) -> bool:
        """
        Returns if the classifier matches the situation.
        :param situation:
        :return:
        """
        return matching(self.condition, situation)

    def does_match_backwards(self, situation: Perception) -> bool:
        """
        Returns if 'situation' is matched by the anticipations.
        This is only the case if the specified conditions that have #-symbols
        in the effect part are also matched!
        :param situation:
        :return:
        """
        p = self.condition.get_backwards_anticipation(situation)
        if self.effect.does_match(situation, p):
            return True
        return False

    def get_best_anticipation(self, perception: Perception) -> Perception:
        """
        Returns the anticipation, the classifier believes to happen most
        probably. This is usually the normal anticipation.
        However, if PEEs are activated, the most probable
        value of each attribute is returned.
        :param perception: Perception
        :return:
        """
        return self.effect.get_best_anticipation(perception)

    def get_backwards_anticipation(self, perception: Perception) \
            -> Optional[Perception]:
        """
        Returns the backwards anticipation.
        Returns -1 if the backwards anticipation was impossible to create.
        This is the case if changing attributes are not specified
        in the conditions.
        :param perception:
        :return:
        """
        back_anticipation = self.condition.\
            get_backwards_anticipation(perception)
        if not self.effect.\
                does_specify_only_changes_backwards(back_anticipation,
                                                    perception):
            # If a specified attribute in the effect part matches
            # the anticipated 'back_anticipation', the backward anticipation
            # fails, because a specified attribute in Effect means a change!
            return None
        return back_anticipation
