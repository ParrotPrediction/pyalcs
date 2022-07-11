from __future__ import annotations
import lcs.agents.acs2 as acs


class Effect(acs.Effect):

    def __init__(self, observation):
        super().__init__(observation)

    @classmethod
    def item_anticipate_change(cls, item, p0_item, p1_item) -> bool:
        if item == cls.WILDCARD or item == '0.0':
            if p0_item != p1_item:
                return False
        else:
            if p0_item == p1_item:
                return False

        return True

