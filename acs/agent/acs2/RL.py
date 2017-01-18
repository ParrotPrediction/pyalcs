import logging

from . import Constants as c

logger = logging.getLogger(__name__)


def apply_rl(match_set: list,
             action_set: list,
             reward: int,
             beta=None,
             gamma=None) -> None:
    """
    Updates classifiers in the action set with given reward.

    :param match_set:
    :param action_set:
    :param reward:
    :param beta:
    :param gamma:
    :return:
    """

    if beta is None:
        beta = c.BETA

    if gamma is None:
        gamma = c.GAMMA

    logger.debug("Applying RL module")

    sf = _sum_fitness_in_match_set(match_set)

    for classifier in action_set:
        classifier.r += beta * (reward + gamma * sf - classifier.r)
        classifier.ir += beta * (reward - classifier.ir)


def _sum_fitness_in_match_set(match_set: list) -> int:
    """
    Returns the sum of fitness value for all classifiers with no generic
    condition part.

    :param match_set: list of classifiers in the match set
    :return: sum of fitness value
    """
    sum_fitness = 0

    for classifier in match_set:
        if classifier.effect != c.CLASSIFIER_WILDCARD * c.CLASSIFIER_LENGTH:
            sum_fitness += classifier.fitness()

    return sum_fitness
