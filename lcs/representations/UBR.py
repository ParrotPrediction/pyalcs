from dataclasses import dataclass


@dataclass
class UBR:
    """
    Real-value representation for unordered-bounded values.
    """
    x1: int
    x2: int

    @property
    def lower_bound(self):
        return min(self.x1, self.x2)

    @property
    def upper_bound(self):
        return max(self.x1, self.x2)

    def contains(self, val: int) -> bool:
        return self.lower_bound < val <= self.upper_bound

    def __hash__(self):
        return hash((self.lower_bound, self.upper_bound))

    def __eq__(self, o) -> bool:
        return self.lower_bound == o.lower_bound \
            and self.upper_bound == o.upper_bound
