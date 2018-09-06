from __future__ import annotations

import random
from typing import Optional, List, Callable

from lcs import Perception
from lcs.representations import UBR
from . import Condition, Effect, Mark, Configuration


class Classifier:

    def __init__(self,
                 condition: Optional[Condition] = None,
                 action: Optional[int] = None,
                 effect: Optional[Effect] = None,
                 quality: float = 0.5,
                 reward: float = 0.5,
                 intermediate_reward: float = 0.0,
                 numerosity: int = 1,
                 experience: int = 1,
                 talp=None,
                 tga: int = 0,
                 tav: float = 0.0,
                 cfg: Optional[Configuration] = None) -> None:

        if cfg is None:
            raise TypeError("Configuration should be passed to Classifier")

        self.cfg = cfg

        def build_condition(initial):
            if initial:
                return Condition(initial, cfg=cfg)

            return Condition.generic(cfg=cfg)

        def build_effect(initial):
            if initial:
                return Effect(initial, cfg=cfg)

            return Effect.pass_through(cfg=cfg)

        self.condition = build_condition(condition)
        self.action = action
        self.effect = build_effect(effect)

        self.mark = Mark(cfg=cfg)
        self.q = quality
        self.r = reward
        self.ir = intermediate_reward
        self.num = numerosity

        self.exp = experience
        self.talp = talp
        self.tga = tga
        self.tav = tav

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
    def specified_unchanging_attributes(self) -> List[int]:
        """
        Determines the number of specified unchanging attributes in
        the classifier. An unchanging attribute is one that is anticipated
        not to change in the effect part.

        Returns
        -------
        List[int]
            list specified unchanging attributes indices
        """
        indices = []

        for idx, (cpi, epi) in enumerate(zip(self.condition, self.effect)):
            if cpi != self.cfg.classifier_wildcard and \
                    epi == self.cfg.classifier_wildcard:
                indices.append(idx)

        return indices

    @property
    def is_subsumer(self) -> bool:
        """
        Determines whether the classifier satisfies the subsumer criteria.

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

    def specialize(self,
                   p0: Perception,
                   p1: Perception,
                   leave_specialized=False) -> None:
        """
        Specializes the effect part where necessary to correctly anticipate
        the changes from p0 to p1 and returns a condition which specifies
        the attributes which must be specified in the condition part.
        The specific attributes in the returned conditions are set to
        the necessary values.

        For real-valued representation a narrow, fixed point UBR is created
        for condition and effect part using the encoded perceptions.

        Parameters
        ----------
        p0: Perception
            previous raw perception obtained from environment
        p1: Perception
            current raw perception obtained from environment
        leave_specialized: bool
            Requires the effect attribute to be a wildcard to specialize it.
            By default false
        """
        p0_enc = list(map(self.cfg.encoder.encode, p0))
        p1_enc = list(map(self.cfg.encoder.encode, p1))

        for idx, item in enumerate(p1_enc):
            if leave_specialized:
                if self.effect[idx] != self.cfg.classifier_wildcard:
                    # If we have a specialized attribute don't change it.
                    continue

            if p0_enc[idx] != p1_enc[idx]:
                self.effect[idx] = UBR(p1_enc[idx], p1_enc[idx])
                self.condition[idx] = UBR(p0_enc[idx], p0_enc[idx])

    def is_reliable(self) -> bool:
        return self.q > self.cfg.theta_r

    def is_inadequate(self) -> bool:
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

    def does_anticipate_change(self) -> bool:
        """
        Checks whether any change in environment is anticipated

        Returns
        -------
        bool
            true if the effect part contains any specified attributes
        """
        return self.effect.specify_change

    def does_anticipate_correctly(self,
                                  previous_situation: Perception,
                                  situation: Perception) -> bool:
        """
        Checks whether classifier correctly performs anticipation.

        Parameters
        ----------
        previous_situation: Perception
            Previously observed perception
        situation: Perception
            Current perception

        Returns
        -------
        bool
            True if anticipation is correct, False otherwise
        """
        p0_enc = list(map(self.cfg.encoder.encode, previous_situation))
        p1_enc = list(map(self.cfg.encoder.encode, situation))

        for idx, eitem in enumerate(self.effect):
            if eitem == self.cfg.classifier_wildcard:
                if p0_enc[idx] != p1_enc[idx]:
                    return False
            else:
                if p0_enc[idx] == p1_enc[idx]:
                    return False

                if p1_enc[idx] not in eitem:
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
            current time step
        """
        # TODO p5: write test
        if 1. / self.exp > self.cfg.beta:
            self.tav = (self.tav * self.exp + (time - self.talp)) / (
                self.exp + 1)
        else:
            self.tav += self.cfg.beta * ((time - self.talp) - self.tav)

        self.talp = time

    def is_more_general(self, other: Classifier) -> bool:
        """
        Checks if the current classifier is more general than the other.
        The average area covered by condition attributes is compared.

        For example UBR(0, 10) is more general than UBR(5, 6) because
        it covers more values.

        Parameters
        ----------
        other: Classifier
            other classifier to compare

        Returns
        -------
        bool
            True is current classifier is more general, False otherwise
        """
        return self.condition.cover_ratio > other.condition.cover_ratio

    def is_marked(self):
        return self.mark.is_marked()

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
        if self.is_subsumer and \
            self.is_more_general(other) and \
            self.condition.does_match_condition(other.condition) and \
                self.effect == other.effect:
            return True

        return False
