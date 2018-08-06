from typing import Optional

from . import Condition, Configuration


class Classifier:

    def __init__(self,
                 condition: Optional[Condition] = None,
                 action: Optional[int] = None,
                 cfg: Optional[Configuration] = None) -> None:

        if cfg is None:
            raise TypeError("Configuration should be passed to Classifier")

        self.cfg = cfg

        def build_condition(initial):
            if initial:
                return Condition(initial, cfg=cfg)

            return Condition.generic(cfg=cfg)

        self.condition = build_condition(condition)
        self.action = action
