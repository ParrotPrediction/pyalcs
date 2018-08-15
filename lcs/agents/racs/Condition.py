from lcs import Perception
from lcs.representations import UBR
from . import Configuration
from .. import PerceptionString
from copy import copy


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

    def specialize(self, idx: int, val: int) -> None:
        """
        Specializes with encoded perception attribute for given index.
        The condition attribute will be assigned a narrow UBR(val, val)
        range.

        Parameters
        ----------
        idx: index of condition attribute
        val: encoded perception value
        """
        # TODO: think - maybe it's useless. In this moment might be used in
        # Mark.get_differences function. I don't like the idea of explicit
        # UBR class here.
        self[idx] = UBR(val, val)

    def does_match(self, perception: Perception):
        encoded_perception = map(self.cfg.encoder.encode, perception)
        return all(p in ubr for p, ubr in zip(encoded_perception, self))
