from copy import copy

from lcs.agents.xncs import Configuration, Effect


class Backpropagation:

    def __init__(self,
                 cfg: Configuration,
                 percentage: float):
        self.percentage = percentage
        self.cfg = cfg
        self.classifiers_for_update = []

    def run_bp(self,
               action_set,
               next_vector: Effect):
        for cl in action_set:
            cl.guesses += 1
            if cl.effect != next_vector:
                cl.mistakes += 1
                if not any(cl == inside[0] for inside in self.classifiers_for_update):
                    self.classifiers_for_update.append(
                        [cl, next_vector, self.cfg.lmc]
                    )
        self.check_if_needed()
        self.update_errors()

    def check_if_needed(self):
        for cl in self.classifiers_for_update:
            if cl[0].effect == cl[1]:
                self.classifiers_for_update.remove(cl)
            else:
                cl[2] -= 1
            if cl[2] == 0:
                self.classifiers_for_update.remove(cl)

    def update_errors(self):
        for cl in self.classifiers_for_update:
            cl[0].error += self.cfg.lem

    def update_effect(self,
                      action_set, next_situation):
        # effect = action_set.fittest_classifier.effect
        effect = Effect(next_situation)
        for cl in action_set.least_fit_classifiers(self.percentage):
            cl.effect = copy(effect)
