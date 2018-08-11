from copy import copy

from lcs import Perception
from . import Configuration
from .. import PerceptionString


class Effect(PerceptionString):
    """
    Anticipates the effects that the classifier 'believes'
    to be caused by the specified action.
    """

    def __init__(self, lst, cfg: Configuration) -> None:
        self.cfg = cfg
        super().__init__(lst, cfg.classifier_wildcard, cfg.oktypes)

    @classmethod
    def pass_through(cls, cfg: Configuration):
        """
        Generates an effect string consisting only of pass-through symbols.

        Parameters
        ----------
        cfg: Configuration
            Configuration of RACS algorithm

        Returns
        -------
        Effect
            Effect list
        """
        ps_str = [copy(cfg.classifier_wildcard) for _
                  in range(cfg.classifier_length)]
        return cls(ps_str, cfg)

    def does_anticipate_correctly(self,
                                  previous_situation: Perception,
                                  situation: Perception) -> bool:
        # TODO: implement
        raise NotImplementedError()

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
        encoded_p0 = list(map(self.cfg.encoder.encode, p0))
        encoded_p1 = list(map(self.cfg.encoder.encode, p1))

        for p0i, p1i, ei in zip(encoded_p0, encoded_p1, self):
            if ei != self.wildcard:
                if p1i not in ei or p0i == p1i:
                    return False

        return True
