from .Configuration import Configuration
from .Condition import Condition
from .Effect import Effect
from .Mark import Mark
from .Classifier import Classifier
from .ClassifiersList import ClassifiersList
from .RACS import RACS

DELTA = 0.001


def eq(p0: float, p1: float) -> bool:
    return abs(p0 - p1) < DELTA
