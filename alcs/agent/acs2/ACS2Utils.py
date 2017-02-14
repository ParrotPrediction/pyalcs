import logging
from random import random, choice

from . import Classifier
from . import Constants as c

logger = logging.getLogger(__name__)


def get_general_perception(dont_care_symbol: str = None,
                           string_length: int = None) -> list:
    """
    Generates a list of general (consisting of wildcards percetion string.
    I.e. ['#', '#', '#']

    :param dont_care_symbol: don't care symbol
    :param string_length: length of the string
    :return: list
    """
    if dont_care_symbol is None:
        dont_care_symbol = c.CLASSIFIER_WILDCARD

    if string_length is None:
        string_length = c.CLASSIFIER_LENGTH

    return [dont_care_symbol] * string_length


def generate_initial_classifiers(number_of_actions: int = None) -> list:
    """
    Generate a list of default, general classifiers for all
    possible actions.

    :param number_of_actions: number of general classifiers
    to be generated
    :return: list of classifiers
    """
    if number_of_actions is None:
        number_of_actions = c.NUMBER_OF_POSSIBLE_ACTIONS

    initial_classifiers = []

    for i in range(number_of_actions):
        cl = Classifier()
        cl.action = i
        cl.t = 0

        initial_classifiers.append(cl)

    logger.debug('Generated initial classifiers: %s', initial_classifiers)
    return initial_classifiers


def generate_match_set(classifiers: list, perception: list) -> list:
    """
    Generates a list of classifiers from population that match given
    perception. A list is then stored
    as agents property.

    :param classifiers: list of classifiers
    :param perception: environment perception as list of values
    :return: a list of matching classifiers
    """
    match_set = []

    for classifier in classifiers:
        if _does_match(classifier, perception):
            match_set.append(classifier)

    logger.debug('Generated match set: [%s]', match_set)
    return match_set


def generate_action_set(classifiers: list, action: int) -> list:
    """
    Generates a list of classifiers with matching action.

    :param classifiers: a list of classifiers (match set)
    :param action: desired action identifier
    :return: a list of classifiers
    """
    action_set = []

    for classifier in classifiers:
        if classifier.action == action:
            action_set.append(classifier)

    logger.debug('Generated action set: [%s]', action_set)
    return action_set


def choose_action(classifiers: list, epsilon=None) -> int:
    """
    Use epsilon-greedy method for action selection. However, it is not clear
    which action is actually the best to choose (since once situation-action
    tuple is mostly represented by several distinct classifiers.

    A random action is selected with epsilon probability. In the other case
    the best classifier (with greatest fitness score).

    :param classifiers: match set
    :param epsilon: probability of returning random action
    :return: an integer representing an action
    """
    if epsilon is None:
        epsilon = c.EPSILON

    if random() < epsilon:
        all_actions = [i for i in range(c.NUMBER_OF_POSSIBLE_ACTIONS)]
        random_action = choice(all_actions)
        logger.debug('Action chosen: [%d] (randomly)', random_action)
        return random_action
    else:
        best_cl = classifiers[0]  # TODO: I would use `choice` here

        for cl in classifiers:
            if (cl.effect != get_general_perception() and
                    cl.fitness() > best_cl.fitness()):
                best_cl = cl

        logger.debug('Action chosen: [%d] (%s)',
                     best_cl.action, best_cl)

        return best_cl.action


def generate_random_int_number(max_value: int) -> int:
    """
    Generates random integer number from 0 to max_value.

    :param max_value: maximum range
    :return: random number
    """
    return int(random() * max_value + 1)


def _does_match(classifier: Classifier, perception: list) -> bool:
    """
    Check if classifier condition match given perception

    :param classifier: classifier object
    :param perception: perception given as list
    :return: True if classifiers can be applied
    for given perception, false otherwise
    """
    if len(perception) != len(classifier.condition):
        raise ValueError('Perception and classifier condition '
                         'length is different')

    for i in range(len(perception)):
        if (classifier.condition[i] != c.CLASSIFIER_WILDCARD and
                classifier.condition[i] != perception[i]):
            return False

    return True


def remove_classifier(classifiers: list, classifier: Classifier) -> None:
    """
    Removes classifier from collection

    :param classifiers: list of classifiers
    :param classifier: classifier to remove
    """
    for cl in classifiers:
        if cl == classifier:
            logger.debug("Removing %s", cl)
            classifiers.remove(cl)
