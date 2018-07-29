from itertools import groupby
from random import random, randint
from typing import Optional

from lcs.agents.acs2 import Classifier, ClassifiersList


def explore(cll: ClassifiersList, pb: float = 0.5) -> Optional[int]:
    """
    Chooses action according to current exploration policy

    Parameters
    ----------
    cll: ClassifiersList
        classifier list
    pb: float
        probability of biased exploration

    Returns
    -------
    int
        action to be executed
    """
    if random() < pb:
        # We are in the biased exploration
        if random() < 0.5:
            return choose_latest_action(cll)
        else:
            return choose_action_from_knowledge_array(cll)

    return choose_random_action(cll)


def exploit(cll: ClassifiersList) -> int:
    """
    Chooses best action according to fitness. If there is no classifier
    in list (or none is predicting change) than a random action is returned

    Parameters
    ----------
    cll: ClassifiersList
        classifier list

    Returns
    -------
    int
        action from the best classifier
    """
    best_classifier = None
    anticipated_change_cls = [cl for cl in cll
                              if cl.does_anticipate_change()]

    if len(anticipated_change_cls) > 0:
        best_classifier = max(anticipated_change_cls,
                              key=lambda cl: cl.fitness)

    if best_classifier is not None:
        return best_classifier.action

    return choose_random_action(cll)


def choose_latest_action(cll: ClassifiersList) -> Optional[int]:
    """
    Chooses latest executed action ("action delay bias")

    Parameters
    ----------
    cll: ClassifiersList
        classifier list

    Returns
    -------
    int
        chosen action number
    """
    last_executed_cls: Classifier
    number_of_cls_per_action = \
        {i: 0 for i in range(cll.cfg.number_of_possible_actions)}

    if len(cll) > 0:
        last_executed_cls = min(cll, key=lambda cl: cl.talp)

        cll.sort(key=lambda cl: cl.action)
        for _action, _clss in groupby(cll, lambda cl: cl.action):
            number_of_cls_per_action[_action] = \
                sum([cl.num for cl in _clss])

    # If there are some actions with no classifiers - select them
    for action, nCls in number_of_cls_per_action.items():
        if nCls == 0:
            return action

    # Otherwise return the action of the last executed classifier
    return last_executed_cls.action


def choose_action_from_knowledge_array(cll: ClassifiersList) -> int:
    """
    Creates 'knowledge array' that represents the average quality of the
    anticipation for each action in the current list. Chosen is
    the action, ACS2 knows least about the consequences.

    Parameters
    ----------
    cll: ClassifiersList
        classifier list

    Returns
    -------
    int
        chosen action
    """
    knowledge_array = {i: 0.0
                       for i in range(cll.cfg.number_of_possible_actions)}

    cll.sort(key=lambda cl: cl.action)

    for _action, _clss in groupby(cll, lambda cl: cl.action):
        _classifiers = [cl for cl in _clss]

        agg_q = sum(cl.q * cl.num for cl in _classifiers)
        agg_num = sum(cl.num for cl in _classifiers)

        knowledge_array[_action] = agg_q / float(agg_num)

    by_quality = sorted(knowledge_array.items(), key=lambda el: el[1])
    action = by_quality[0][0]

    return action


def choose_random_action(cll: ClassifiersList) -> int:
    """
    Chooses one of the possible actions in the environment randomly

    Parameters
    ----------
    cll: ClassifiersList
        classifier list

    Returns
    -------
    int
        random action number

    """
    return randint(0, cll.cfg.number_of_possible_actions - 1)
