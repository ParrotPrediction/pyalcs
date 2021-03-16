from __future__ import annotations

from .. import ImmutableSequence


class Condition(ImmutableSequence):

    def subsumes(self, other) -> bool:
        for ci, oi in zip(self, other):
            if ci != self.WILDCARD and oi != self.WILDCARD and ci != oi:
                return False
        return True

    @property
    def wildcard_number(self) -> int:
        return sum(1 for c in self if c == self.WILDCARD)

    def is_more_general(self, other: Condition) -> bool:
        for ci, oi in zip(self, other):
            if ci != self.WILDCARD and ci != oi:
                return False
        return True
