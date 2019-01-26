from __future__ import annotations


def _assert_proper_value(val: float):
    assert type(val) is float
    if val < 0.0 or val > 1.0:
        raise ValueError(f'Value {val:.2f} not in range [0, 1]')


class Interval:
    """
    Representation of interval for real-valued boundaries. Can work
    for both OBR (ordered-bounded representation) and UBR (unordered-bounded
    representation).
    """

    __slots__ = ['x1', 'x2']

    def __init__(self, x1: float, x2: float) -> None:
        _assert_proper_value(x1)
        _assert_proper_value(x2)

        self.x1 = x1
        self.x2 = x2

    @property
    def left_bound(self) -> float:
        return min(self.x1, self.x2)

    @property
    def right_bound(self) -> float:
        return max(self.x1, self.x2)

    @property
    def span(self) -> float:
        return self.right_bound - self.left_bound

    def __contains__(self, item):
        assert type(item) in [Interval, float]

        if type(item) is Interval:
            return self.left_bound <= item.left_bound and \
                self.right_bound >= item.right_bound

        elif type(item) is float:
            return self.left_bound <= item <= self.right_bound

        else:
            return False

    def __hash__(self):
        return hash((self.left_bound, self.right_bound))

    def __eq__(self, o) -> bool:
        delta_left = abs(self.left_bound - o.left_bound)
        delta_right = abs(self.right_bound - o.right_bound)
        return (delta_left + delta_right) < 0.01

    def __repr__(self):
        return f"[{self.left_bound:.2f};{self.right_bound:.2f}]"
