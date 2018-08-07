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
        pass
