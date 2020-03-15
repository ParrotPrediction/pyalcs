from itertools import groupby

import numpy as np

import lcs.agents.acs as acs
import lcs.agents.acs2 as acs2
from . import RandomAction, BestAction


class KnowledgeArray:
    """
    Knowledge-array bias.

    Creates 'knowledge array' that represents the average quality of the
    anticipation for each action in the current list. Chosen is
    the action, ACS2 knows least about the consequences.
    """

    def __init__(self, all_actions: int, **kwargs):
        self.all_actions = all_actions
        self.epsilon = kwargs['epsilon']
        self.biased_exploration_prob = kwargs['biased_exploration_prob']
        assert 0 <= self.epsilon < 1
        assert 0 <= self.biased_exploration_prob < 1

    def __call__(self, population) -> int:
        if np.random.rand() < self.epsilon:
            if np.random.rand() < self.biased_exploration_prob:
                # We are in biased exploration
                if type(population) is acs.ClassifiersList:
                    raise NotImplementedError()
                elif type(population) is acs2.ClassifiersList:
                    return self._handle_knowledge_array_for_acs2(population)
                else:
                    raise TypeError()
            else:
                # Normal exploration
                return RandomAction(all_actions=self.all_actions)(population)
        else:
            return BestAction(all_actions=self.all_actions)(population)

    def _handle_knowledge_array_for_acs2(self, population) -> int:
        knowledge_array = {i: 0.0 for i in range(self.all_actions)}

        population.sort(key=lambda cl: cl.action)

        for _action, _clss in groupby(population, lambda cl: cl.action):
            _classifiers = [cl for cl in _clss]

            agg_q = sum(cl.q * cl.num for cl in _classifiers)
            agg_num = sum(cl.num for cl in _classifiers)

            knowledge_array[_action] = agg_q / float(agg_num)

        by_quality = sorted(knowledge_array.items(), key=lambda el: el[1])
        action = by_quality[0][0]

        return action
