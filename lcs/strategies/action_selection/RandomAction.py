import numpy as np


class RandomAction:

    def __init__(self, all_actions: int):
        self.all_actions = all_actions

    def __call__(self, population):
        return np.random.randint(self.all_actions)
