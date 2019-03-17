import numpy as np

from .utils import check_types
from .Perception import Perception
from .TypedList import TypedList

# Tolerance for comparing two real numbers
DELTA = 0.05


def is_different(a: float, b: float):
    return abs(a - b) > DELTA


def clip(val: float):
    return np.clip(val, 0, 1).tolist()
