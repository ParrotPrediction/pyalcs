from copy import copy

from lcs import Perception
from . import Configuration
from .. import PerceptionString


class Condition(PerceptionString):

    def __init__(self, lst, cfg: Configuration) -> None:
        self.cfg = cfg
        super().__init__(lst, cfg.classifier_wildcard, cfg.oktypes)

    @classmethod
    def generic(cls, cfg: Configuration):
        ps_str = [copy(cfg.classifier_wildcard) for _
                  in range(cfg.classifier_length)]
        return cls(ps_str, cfg)

    @property
    def specificity(self) -> int:
        """
        Returns
        -------
        int
            Number of not generic (wildcards) attributes
        """
        return sum(1 for c in self if c != self.wildcard)

    def generalize(self, idx: int):
        self[idx] = self.cfg.classifier_wildcard

    def does_match(self, perception: Perception):
        encoded_perception = map(self.cfg.encoder.encode, perception)
        return all(p in ubr for p, ubr in zip(encoded_perception, self))
