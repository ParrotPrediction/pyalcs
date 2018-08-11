from lcs import Perception
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

    def does_match(self, perception: Perception):
        encoded_perception = map(self.cfg.encoder.encode, perception)
        return all(p in ubr for p, ubr in zip(encoded_perception, self))
