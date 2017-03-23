import logging

from . import Constants as c
from . import ACS2Utils

logger = logging.getLogger(__name__)


def apply_rl(match_set: list,
             action_set: list,
             reward: int,
             learning_rate: float,
             discount_factor: float) -> None:
    """
    Updates classifiers reward and intermediate reward in the action set
    with given obtained reward following the idea of Q-learning.

    To guarantee that procedure works successfully, it's mandatory that
    the model is specific enough.

    Function updates properties for *all* classifiers stored in the action set.

    :param match_set: match set (list of classifiers)
    :param action_set: action set (list of classifiers), will be changed
    :param reward: intermediate reward obtained from the environment
    :param learning_rate:
    :param discount_factor:
    """
    logger.info("Applying RL module, distributing reward: [%d]", reward)

    max_p = _calculate_maximum_payoff(match_set)

    for cl in action_set:
        cl.r += learning_rate * (reward + discount_factor * max_p - cl.r)
        cl.ir += learning_rate * (reward - cl.ir)
        logger.info("\t\t[RL] Classifier: %s", cl)


# TODO: I'm not sure about this part
def _calculate_maximum_payoff(match_set: list) -> float:
    """
    Calculate the maximum payoff predicted in the next time-step

    :param match_set: list of classifiers (match set)
    :return: maximum fitness value found in match_set (maxP)
    """
    wildcards = ACS2Utils.get_general_perception()
    applicable_classifiers = [cl for cl in match_set if cl.effect != wildcards]

    if len(applicable_classifiers) > 0:
        return max(cl.fitness() for cl in applicable_classifiers)

    return 0
