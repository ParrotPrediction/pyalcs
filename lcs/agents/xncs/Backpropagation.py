from copy import copy

from lcs.agents.xncs import Configuration, Effect


class Backpropagation:

    def __init__(self,
                 cfg: Configuration,
                 percentage: float):
        self.percentage = percentage
        self.cfg = cfg
        self.update_cycles = 0

    def update_cycle(self,
                     action_set,
                     update_vector: Effect):
        if action_set.fittest_classifier.effect != update_vector:
            self.update_cycles = self.cfg.lmc

        if self.update_cycles > 0:
            if action_set.fittest_classifier.effect == update_vector:
                self.update_cycles = 0
            else:
                self.update_cycles -= 1
                self._update_classifiers_effect(
                    action_set.least_fit_classifiers(self.percentage),
                    update_vector
                )
                self._update_classifiers_error(
                    action_set,
                    update_vector
                )

    def _update_classifiers_effect(self, classifiers_for_update, update_vector):
        for cl in classifiers_for_update:
            if cl.effect != update_vector:
                cl.effect = copy(update_vector)

    def _update_classifiers_error(self, classifiers_for_update, update_vector):
        for cl in classifiers_for_update:
            cl.queses += 1
            if cl.effect != update_vector:
                cl.mistakes += 1
                cl.error += self.cfg.lem * cl.error
