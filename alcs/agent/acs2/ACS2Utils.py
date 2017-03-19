import logging
from random import random

from alcs.strategies.ActionSelection import ActionSelection, BestAction
from . import Classifier
from . import Constants as c

logger = logging.getLogger(__name__)


def get_general_perception(dont_care_symbol: str = None,
                           string_length: int = None) -> list:
    """
    Generates a list of general (consisting of wildcards perception string.
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


def choose_action(classifiers: list,
                  epsilon: float,
                  strategy: ActionSelection) -> int:
    """
    Model exploration/exploitation mechanism. For exploration
    phase a custom strategy is evaluated with given probability.

    :param classifiers: match set
    :param epsilon: probability of executing exploration path
    :param strategy: custom strategy used for exploration
    :return: an integer representing an action
    """
    if random() < epsilon:
        # Exploration phase
        return strategy.select_action(classifiers)
    else:
        # Exploitation phase - take the best possible classifier
        return BestAction().select_action(classifiers)


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


def does_anticipate_correctly(classifier: Classifier,
                              perception: list,
                              previous_perception: list) -> bool:
    """
    Checks anticipation. While the pass-through symbols in the effect part
    of a classifier directly anticipate that these attributes stay the same
    after the execution of an action, the specified attributes anticipate
    a change to the specified value. Thus, if the perceived value did not
    change to the anticipated but actually stayed at the value, the classifier
    anticipates incorrectly.

    :param classifier: given classifier
    :param perception: current perception
    :param previous_perception: previous perception
    :return: True if classifier anticipates correctly, False otherwise
    """
    for i in range(c.CLASSIFIER_LENGTH):
        if classifier.effect[i] == c.CLASSIFIER_WILDCARD:
            if previous_perception[i] != perception[i]:
                return False
        else:
            if (classifier.effect[i] != perception[i] or
                    previous_perception[i] == perception[i]):
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
            logger.info("Removing %s", cl)
            classifiers.remove(cl)
