from typing import Tuple


def visualize(interval: Tuple[int, int],
              val_range: Tuple[int, int],
              n: int = 10):

    range_len = len(range(val_range[0], val_range[1])) + 1

    if range_len < n:
        n = range_len

    lower = _scale(interval[0], range_len, n)
    upper = _scale(interval[1], range_len, n)

    visualization = '.' * n
    selected = 'O' * (len(range(lower, upper)) + 1)

    return visualization[:lower] + selected + visualization[upper + 1:]


def _scale(val: int, max_range: int, n: int) -> int:
    return round(n * val / max_range)
