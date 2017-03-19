import logging
from abc import ABCMeta, abstractmethod
from itertools import groupby
from random import choice

from alcs.agent.acs2 import Constants as c

logger = logging.getLogger(__name__)


class ActionSelection(metaclass=ABCMeta):
    @abstractmethod
    def select_action(self, classifiers: list) -> int:
        raise NotImplementedError


class Greedy(ActionSelection):
    """
    Randomly selects an action from given classifiers.
    """
    def select_action(self, classifiers: list) -> int:
        matchset_actions = {cl.action for cl in classifiers}
        random_action = choice(list(matchset_actions))

        logger.debug('Action chosen: [%d] (greedy)', random_action)

        return random_action


class BestAction(ActionSelection):
    """
    Choose best action from population
    """
    def select_action(self, classifiers: list) -> int:
        from alcs.agent.acs2.ACS2Utils import get_general_perception

        best_cl = classifiers[0]

        for cl in classifiers:
            if (cl.effect != get_general_perception() and
                        cl.fitness() > best_cl.fitness()):
                best_cl = cl

        action = best_cl.action
        logger.info('Best action chosen: [%d] (%s)', action, best_cl)

        return action


class ActionDelayBias(ActionSelection):
    def select_action(self, classifiers: list) -> int:
        all_actions = {i for i in range(c.NUMBER_OF_POSSIBLE_ACTIONS)}
        matchset_actions = {cl.action for cl in classifiers}

        old_actions = all_actions.difference(matchset_actions)

        if len(old_actions) > 0:
            action = choice(list(old_actions))
        else:
            oldest_cls = min(classifiers, key=lambda cls: cls.t_alp)
            action = oldest_cls.action

        logger.info('Action chosen: [%d] (action delay bias)', action)

        return action


class KnowledgeArrayBias(ActionSelection):
    def select_action(self, classifiers: list) -> int:
        knowledge_array = {i: 0 for i in range(c.NUMBER_OF_POSSIBLE_ACTIONS)}
        classifiers.sort(key=lambda cl: cl.action)

        for _action, _clss in groupby(classifiers, lambda cl: cl.action):
            _classifiers = [cl for cl in _clss]

            agg_q = sum(cl.q * cl.num for cl in _classifiers)
            agg_num = sum(cl.num for cl in _classifiers)

            knowledge_array[_action] = agg_q / float(agg_num)

        by_quality = sorted(knowledge_array.items(), key=lambda el: el[1])

        action = by_quality[0][0]

        logger.info('Action chosen: [%d] (knowledge array bias)', action)

        return action
