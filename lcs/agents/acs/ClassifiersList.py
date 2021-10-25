from __future__ import annotations

from typing import Optional

from lcs import TypedList, Perception
from lcs.agents.acs import Classifier


class ClassifiersList(TypedList):
    """
    Represents overall population, match/action sets
    """

    def __init__(self, *args, oktypes=(Classifier,)) -> None:
        super().__init__(*args, oktypes=oktypes)

    def form_match_set(self, situation: Perception) -> ClassifiersList:
        matching_ls = [cl for cl in self if cl.does_match(situation)]
        return ClassifiersList(*matching_ls)

    def get_maximum_fitness(self) -> float:
        """
        Returns the maximum fitness value amongst those classifiers
        that anticipated a change in environment.

        Returns
        -------
        float
            fitness value
        """
        anticipated_change_cls = [cl for cl in self
                                 if cl.does_anticipate_change()]

        if len(anticipated_change_cls) > 0:
            best_cl = max(anticipated_change_cls, key=lambda cl: cl.fitness)
            return best_cl.fitness

        return 0.0

    def get_best_classifier(self) -> Optional[Classifier]:
        return max(self, key=lambda cl: cl.fitness)

    def __str__(self):
        return "\n".join(str(classifier)
                         for classifier
                         in sorted(self, key=lambda cl: -cl.fitness))
