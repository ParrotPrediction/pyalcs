from functools import partial
from typing import Tuple


def _scale(val: int, max_range: int, n: int) -> int:
    return round(n * val / max_range)


def visualize(interval: Tuple[int, int],
              val_range: Tuple[int, int],
              buckets: int = 10):

    def select_value(current, interval, mapped):
        p = mapped[interval[0]]
        q = mapped[interval[1]]

        if current < p or current > q:
            return '.'

        if current in [p, q]:
            # edges - process possible ambiguities
            if len([v for v in mapped.values() if v == current]) > 1:
                return 'o'

        return 'O'

    mapped = {val: _scale(val, max_range=val_range[1] + 1, n=buckets) for
              val in range(val_range[0], val_range[1] + 1)}
    rep = map(partial(select_value, interval=interval, mapped=mapped),
              range(0, buckets))
    return "".join(rep)
