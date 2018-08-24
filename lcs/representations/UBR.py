from dataclasses import dataclass


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
        return sum(1 for _ in range(self.lower_bound, self.upper_bound))

    def __contains__(self, item):
        return self.lower_bound <= item <= self.upper_bound

    def __hash__(self):
        return hash((self.lower_bound, self.upper_bound))

    def __eq__(self, o) -> bool:
        return self.lower_bound == o.lower_bound \
            and self.upper_bound == o.upper_bound
