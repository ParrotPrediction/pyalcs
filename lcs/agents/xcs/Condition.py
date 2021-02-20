from __future__ import annotations

from .. import ImmutableSequence


class Condition(ImmutableSequence):

    def subsumes(self, other) -> bool:
        for ci, oi in zip(self, other):
            if ci != self.WILDCARD and oi != self.WILDCARD and ci != oi:
                return False
        return True

