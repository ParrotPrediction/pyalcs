from __future__ import annotations

from lcs import TypedList, Perception
from . import Classifier


class ClassifierList(TypedList):

    def __init__(self, *args) -> None:
        super().__init__((Classifier,), *args)

    @classmethod
    def form_match_set(cls,
                       population: ClassifierList,
                       situation: Perception,
                       cfg) -> ClassifierList:
        return cls([cl for cl in population
                    if cl.condition.does_match(situation)], cfg)
