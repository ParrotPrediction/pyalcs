from typing import Optional

from lcs import Perception
from . import Condition, Effect, Mark, Configuration


class Classifier:

    def __init__(self,
                 condition: Optional[Condition]=None,
                 action: Optional[int]=None,
                 effect: Optional[Effect]=None,
                 quality: float=0.5,
                 experience: int=1,
                 talp=None,
                 tav: float=0.0,
                 cfg: Optional[Configuration]=None) -> None:

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

        self.exp = experience
        self.talp = talp
        self.tav = tav

    def increase_experience(self) -> int:
        self.exp += 1
        return self.exp

    def decrease_quality(self) -> float:
        self.q -= self.cfg.beta * self.q
        return self.q

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

        # FIXME: Think - in this proposition there is no idea of an wildcard.
        # However it works fine. Will see later. There might be a problem
        # that for very wide Effect (wildcard) everything will be accepted as
        # correctly anticipated. So some specialization pressure needs to be
        # applied

        for idx, eitem in enumerate(self.effect):
            if eitem.contains(p1_enc[idx]):
                if p0_enc[idx] == p1_enc[idx]:
                    pass
            else:
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
