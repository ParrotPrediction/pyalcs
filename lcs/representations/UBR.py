from __future__ import annotations

from dataclasses import dataclass

from dataslots import with_slots


@with_slots
@dataclass
class UBR:
    """
    Real-value representation for unordered-bounded values.
    """
    x1: float
    x2: float

    @property
    def lower_bound(self) -> float:
        return min(self.x1, self.x2)

    @property
    def upper_bound(self) -> float:
        return max(self.x1, self.x2)

    @property
    def bound_span(self) -> float:
        return self.upper_bound - self.lower_bound

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

    def __contains__(self, item):
        return self.lower_bound <= item <= self.upper_bound

    def __hash__(self):
        return hash((self.lower_bound, self.upper_bound))

    def __eq__(self, o) -> bool:
        return self.lower_bound == o.lower_bound \
            and self.upper_bound == o.upper_bound
