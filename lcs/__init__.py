from .utils import check_types
from .Perception import Perception
from .TypedList import TypedList

# Tolerance for comparing two real numbers
DELTA = 0.01


def is_different(a: float, b: float):
    return abs(a - b) > DELTA
