from lcs.agents.xcs import GeneticAlgorithm as XCSGeneticAlgorithm
from copy import copy
from lcs.agents.xncs import Classifier, ClassifiersList, Configuration


class GeneticAlgorithm(XCSGeneticAlgorithm):

    def __init__(
            self,
            population: ClassifiersList,
            cfg: Configuration
            ):

        super().__init__(population, cfg)

    def _make_children(self, parent1, parent2, time_stamp):
        assert isinstance(parent1, Classifier)
        assert isinstance(parent2, Classifier)

        child1 = Classifier(
            self.cfg,
            copy(parent1.condition),
            copy(parent1.action),
            time_stamp,
            effect=None
        )

        child2 = Classifier(
            self.cfg,
            copy(parent2.condition),
            copy(parent2.action),
            time_stamp,
            effect=None
        )

        return child1, child2
