from __future__ import annotations

from .. import ImmutableSequence


class Condition(ImmutableSequence):

    def subsumes(self, other) -> bool:
        for ci, oi in zip(self, other):
            if ci != self.WILDCARD and oi != self.WILDCARD and ci != oi:
                return False
        return True

    def wildcard_number(self) -> int:
        number_of_wildcard = 0
        for ci in self:
            if ci == self.WILDCARD:
                number_of_wildcard += 1
        return number_of_wildcard

    def is_more_general(self, other: Condition) -> bool:
        for ci, oi in zip(self, other):
            if ci != self.WILDCARD and ci != oi:
                return False
        return True
