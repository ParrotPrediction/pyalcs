import numpy as np


class BestAction:

    def __init__(self, all_actions: int):
        self.all_actions = all_actions

    def __call__(self, population):
        cl = population.get_best_classifier()

        if cl is not None:
            return cl.action
        else:
            return np.random.randint(self.all_actions)
