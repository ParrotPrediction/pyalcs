from itertools import groupby

import numpy as np

import lcs.agents.acs as acs
import lcs.agents.acs2 as acs2
from . import RandomAction, BestAction


class ActionDelay:
    """
    Action-delay bias.
    Chooses action that was executed a longest time ago.
    """

    def __init__(self, all_actions: int, **kwargs):
        self.all_actions = all_actions
        self.epsilon = kwargs['epsilon']
        self.biased_exploration_prob = kwargs['biased_exploration_prob']
        assert 0 <= self.epsilon <= 1
        assert 0 <= self.biased_exploration_prob <= 1

    def __call__(self, population) -> int:
        if np.random.rand() < self.epsilon:
            if np.random.rand() < self.biased_exploration_prob:
                # We are in biased exploration
                if type(population) is acs.ClassifiersList:
                    raise NotImplementedError()
                elif type(population) is acs2.ClassifiersList:
                    return self._handle_latest_action_for_acs2(population)
                else:
                    raise TypeError()
            else:
                # Normal exploration
                return RandomAction(all_actions=self.all_actions)(population)
        else:
            return BestAction(all_actions=self.all_actions)(population)

    def _handle_latest_action_for_acs2(self, population) -> int:
        last_executed_cls = None
        number_of_cls_per_action = {i: 0 for i in range(self.all_actions)}

        if len(population) > 0:
            last_executed_cls = min(population, key=lambda cl: cl.talp)

            # Count how many classifiers are there for each action
            population.sort(key=lambda cl: cl.action)
            for _action, _clss in groupby(population, lambda cl: cl.action):
                number_of_cls_per_action[_action] = \
                    sum([cl.num for cl in _clss])

        # If there are some actions with no classifiers - select them
        for action, nCls in number_of_cls_per_action.items():
            if nCls == 0:
                return action

        # Otherwise return the action of the last executed classifier
        if last_executed_cls:
            return last_executed_cls.action

        # if there is no classifiers at all - select random action
        return RandomAction(all_actions=self.all_actions)(population)
