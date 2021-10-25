import numpy as np

from lcs.strategies.action_selection.BestAction import BestAction
from lcs.strategies.action_selection.RandomAction import RandomAction


class EpsilonGreedy:

    def __init__(self, all_actions: int, **kwargs):
        self.all_actions = all_actions
        self.epsilon = kwargs['epsilon']
        assert 0 <= self.epsilon <= 1

    def __call__(self, population) -> int:
        if np.random.uniform() > self.epsilon:
            return BestAction(all_actions=self.all_actions)(population)
        else:
            return RandomAction(all_actions=self.all_actions)(population)
