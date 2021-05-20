from copy import copy

from lcs.agents.xncs import Configuration, Classifier, ClassifiersList, Effect


class Backpropagation:

    def __init__(self, cfg: Configuration,
                 population: ClassifiersList):
        self.population = population
        self.classifiers_for_update = []
        self.update_vectors = []
        self.cfg = cfg
        self.update_cycles = 0

    def insert_into_bp(self,
                       classifier: Classifier,
                       update_vector: Effect):
        self.classifiers_for_update.append(classifier)
        self.update_vectors.append(update_vector)
        self._check_for_update()

    def _check_for_update(self):
        self.update_cycles += 1
        if self.update_cycles >= self.cfg.lmc:
            self._update_bp()
        # TODO: One of them is NoneType
        elif self.classifiers_for_update[-1].effect == self.update_vectors[-1]:
            self._update_bp()

    def _update_bp(self):
        for cl, uv in zip(self.classifiers_for_update, self.update_vectors):
            if cl.effect != uv:
                cl.effect = copy(uv)
                cl.error += self.cfg.lem * cl.error
        self.classifiers_for_update = []
        self.update_vectors = []
        self.update_cycles = 0

