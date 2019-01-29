from __future__ import annotations

import random
from typing import Optional, List, Callable, Dict

import numpy as np

from lcs import Perception, is_different, clip
from lcs.representations import Interval, FULL_INTERVAL

from . import Condition, Effect, Mark, Configuration


class Classifier:

    __slots__ = ['condition', 'action', 'effect', 'mark', 'q', 'r',
                 'ir', 'num', 'exp', 'talp', 'tga', 'tav', 'ee', 'cfg']

    def __init__(self,
                 condition: Optional[Condition] = None,
                 action: Optional[int] = None,
                 effect: Optional[Effect] = None,
                 quality: float = 0.5,
                 reward: float = 0.5,
                 immediate_reward: float = 0.0,
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
        self.ir = immediate_reward
        self.num = numerosity

        self.exp = experience
        self.talp = talp
        self.tga = tga
        self.tav = tav

        self.ee = False

    def __eq__(self, other):
        # TODO: here we should base on intervals somehow
        if self.condition == other.condition and \
                self.action == other.action and \
                self.effect == other.effect:
            return True

        return False

    def __hash__(self):
        return hash((str(self.condition), self.action, str(self.effect)))

    def __repr__(self):
        return "{}\t{}\t{} x {} fitness: {:.2f}".format(
            self.condition, self.action, self.effect, self.num, self.fitness)

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
            immediate_reward=old_cls.ir,
            cfg=old_cls.cfg)

        new_cls.tga = time
        new_cls.talp = time
        new_cls.tav = old_cls.tav

        return new_cls

    @property
    def fitness(self):
        return self.q * self.r

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

    def specialize(self,
                   p0: Perception,
                   p1: Perception,
                   leave_specialized: bool = False) -> None:
        """
        Specializes the effect part where necessary to correctly anticipate
        the changes from p0 to p1 and returns a condition which specifies
        the attributes which must be specified in the condition part.
        The specific attributes in the returned conditions are set to
        the necessary values.

        For real-valued representation a random noise might be added to both
        `p0` and `p1` (see `Configuration`, `cover_noise` parameter).

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

        for idx, item in enumerate(p1):
            if leave_specialized:
                if self.effect[idx] != self.cfg.classifier_wildcard:
                    # If we have a specialized attribute don't change it.
                    continue

            if is_different(p0[idx], p1[idx]):
                noise = np.random.uniform(0, self.cfg.cover_noise)
                self.condition[idx] = Interval(
                    clip(p0[idx] - noise),
                    clip(p0[idx] + noise))
                self.effect[idx] = Interval(
                    clip(p1[idx] - noise),
                    clip(p1[idx] + noise))

    def is_reliable(self) -> bool:
        return self.q > self.cfg.theta_r

    def is_inadequate(self) -> bool:
        return self.q < self.cfg.theta_i

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
                                  p0: Perception,
                                  p1: Perception) -> bool:
        """
        Checks whether classifier correctly performs anticipation.

        Parameters
        ----------
        p0: Perception
            Previously observed perception
        p1: Perception
            Current perception

        Returns
        -------
        bool
            True if anticipation is correct, False otherwise
        """
        for idx, eitem in enumerate(self.effect):
            if eitem == self.cfg.classifier_wildcard:
                if is_different(p0[idx], p1[idx]):
                    return False
            else:
                if not is_different(p0[idx], p1[idx]):
                    return False

                if p1[idx] not in eitem:
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
            self.ee = False

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

        For example interval [0, 10] is more general than [5, 6] because
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

    def get_interval_proportions(self) -> Dict[int, int]:
        """
        Returns a region to which classifier is assigned based on the
        classifier condition.

        See "For Real! XCS with Continuous-Valued Inputs" by C. Stone and
        L. Bull for a reference.

        Returns
        -------
        Dict[int, int]
            A dictionary with interval region counts within condition part
        """
        counts = {1: 0, 2: 0, 3: 0, 4: 0}

        fl = FULL_INTERVAL

        for i in self.condition:
            if i.left != fl.left and i.right != fl.right:
                counts[1] += 1

            if i.left == fl.left and i.right != fl.right:
                counts[2] += 1

            if i.left != fl.left and i.right == fl.right:
                counts[3] += 1

            if i.left == fl.left and i.right == fl.right:
                counts[4] += 1

        return counts

    def generalize_unchanging_condition_attribute(
            self, randomfunc: Callable = random.choice) -> bool:
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
        # TODO: this might be too aggressive. Consider slight mutation
        if len(self.specified_unchanging_attributes) > 0:
            ridx = randomfunc(self.specified_unchanging_attributes)
            self.condition.generalize(ridx)
            return True

        return False
