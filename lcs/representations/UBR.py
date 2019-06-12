from __future__ import annotations

from dataclasses import dataclass

from dataslots import with_slots


@with_slots
@dataclass
class UBR:
    """
    Real-value representation for unordered-bounded values.
    """
    x1: int
    x2: int

    @property
    def lower_bound(self) -> int:
        return min(self.x1, self.x2)

    @property
    def upper_bound(self) -> int:
        return max(self.x1, self.x2)

    @property
    def bound_span(self) -> int:
        return sum(1 for _ in range(self.lower_bound, self.upper_bound + 1))

    def incorporates(self, other: UBR) -> bool:
        """
        Checks whether current UBR incorporates other.

        Parameters
        ----------
        other: UBR
            Other real-value representation

        Returns
        -------
        bool
            True if `other` is contained in this UBR, False otherwise
        """
        return self.lower_bound <= other.lower_bound and \
            self.upper_bound >= other.upper_bound

    def can_be_merged(self, other: UBR) -> bool:
        if self.lower_bound in [other.lower_bound, other.upper_bound]:
            return True

        if self.upper_bound in [other.lower_bound, other.upper_bound]:
            return True

        return False

    def __contains__(self, item):
        return self.lower_bound <= item <= self.upper_bound

    def __hash__(self):
        return hash((self.lower_bound, self.upper_bound))

    def __eq__(self, o) -> bool:
        return self.lower_bound == o.lower_bound \
            and self.upper_bound == o.upper_bound
