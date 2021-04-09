from copy import copy

from lcs.agents.xncs import Configuration, Classifier, Effect

class Backpropagation:

    def __init__(self, cfg: Configuration):
        self.classifiers_for_update = []
        self.update_vectors = []
        self.cfg = cfg
        self.update_cycles = 0

    def insert_into_bp(self,
                       classifier: Classifier,
                       update_vector: Effect):
        self.classifiers_for_update.append(classifier)
        self.update_vectors.append(update_vector)

    def update_bp(self):
        for cl, uv in zip(self.classifiers_for_update, self.update_vectors):
            if cl.effect is None:
                cl.effect = copy(uv)
            else:
                cl.effect = copy(uv)
                cl.error = cl.error + self.cfg.lem
        self.classifiers_for_update = []
        self.update_vectors = []
        self.update_cycles = 0

    def check_and_update(self):
        self.update_cycles += 1
        if self.update_cycles >= self.cfg.lmc:
            self.update_bp()
