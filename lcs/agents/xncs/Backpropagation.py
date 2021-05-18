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

    def update_bp(self):
        for cl, uv in zip(self.classifiers_for_update, self.update_vectors):
            if cl.effect is None:
                cl.effect = copy(Effect(uv))
                self.remove_copies(cl)
            else:
                cl.effect = copy(Effect(uv))
                pop_cl = self.remove_copies(cl)
                if pop_cl is None:
                    cl.error += self.cfg.lem
                else:
                    pop_cl.error += self.cfg.lem
        self.classifiers_for_update = []
        self.update_vectors = []
        self.update_cycles = 0

    def check_and_update(self):
        self.update_cycles += 1
        if self.update_cycles >= self.cfg.lmc:
            self.update_bp()

    def remove_copies(self, bp_cl):
        for pop_cl in self.population:
            if pop_cl == bp_cl:
                self.population.remove(pop_cl)
                return pop_cl
        return None
