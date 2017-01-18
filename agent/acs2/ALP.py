from agent.acs2.Classifier import Classifier
from agent.acs2 import Constants as c
from agent.acs2.ACS2Utils import remove

from random import random
import logging

logger = logging.getLogger(__name__)


def apply_alp(classifiers: list,
              action: int,
              time: int,
              action_set: list,
              perception: list,
              previous_perception: list):

    logger.debug('Applying ALP module')
    was_expected_case = 0

    for classifier in action_set:
        classifier.exp += 1
        _update_application_average(classifier, time)

        if _does_anticipate_correctly(classifier,
                                      perception,
                                      previous_perception):
            new_classifier = _expected_case(classifier, perception)
            was_expected_case += 1
        else:
            new_classifier = _unexpected_case(classifier,
                                              perception,
                                              previous_perception)
            if classifier.q < c.THETA_I:
                remove(classifier, classifiers)
                action_set.remove(classifier)

        if new_classifier is not None:
            new_classifier.tga = time
            _add_alp_classifier(new_classifier,
                                classifiers,
                                action_set)

    if was_expected_case == 0:
        new_classifier = _cover_triple(previous_perception,
                                       perception,
                                       action,
                                       time)
        _add_alp_classifier(new_classifier, classifiers, action_set)


def _expected_case(classifier: Classifier,
                   perception: list) -> Classifier:

    diff = _get_differences(classifier.mark, perception)

    if diff == [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH:
        classifier.q += c.BETA * (1 - classifier.q)
        return None
    else:
        class_spec_num = _number_of_spec(classifier.condition)
        diff_spec_num = _number_of_spec(diff)
        child = Classifier.copy_from(classifier)

        if class_spec_num == c.U_MAX:
            _remove_random_spec_element(child.condition)
            class_spec_num -= 1

            while class_spec_num + diff_spec_num > c.BETA:
                if class_spec_num > 0 and random() < 0.5:
                    _remove_random_spec_element(child.condition)
                    class_spec_num -= 1
                else:
                    _remove_random_spec_element(diff)
                    diff_spec_num -= 1
        else:
            while class_spec_num + diff_spec_num > c.BETA:
                _remove_random_spec_element(diff)
                diff_spec_num -= 1

        child.condition = diff
        child.exp = 1

        if child.q < 0.5:
            child.q = 0.5

        return child


def _get_differences(mark: list, perception: list) -> list:
    diff = [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH

    if mark is not None:
        type1 = 0  # counts when mark is different from perception
        type2 = 0  # counts when mark is applied

        for i in range(len(perception)):
            if mark[i] != perception[i]:
                type1 += 1
            if int(mark[i]) > 1:
                type2 += 1

        if type1 > 0:
            type1 = random() * type1
            for i in range(len(perception)):
                if mark[i] != perception[i]:
                    if int(type1) == 0:
                        diff[i] = perception[i]
                    type1 -= 1
        elif type2 > 0:
            for i in range(len(perception)):
                if mark[i] != perception[i]:
                    diff[i] = perception[i]

    return diff


def _number_of_spec(condition: list) -> int:
    """
    Returns the number of non-#
    :param condition: a classifier condition or difference list
    :return: number of non-general elements
    """
    n = 0

    for i in range(len(condition)):
        if condition[i] != c.CLASSIFIER_WILDCARD:
            n += 1

    return n


def _remove_random_spec_element(condition: list) -> None:
    """
    Iterates over list and tries to replace random element
    with wildcard symbol.

    Argument is modified.

    :param condition: a classifier condition or a difference list
    """
    searching = True

    while searching:
        for i in condition:
            if i != c.CLASSIFIER_WILDCARD and random() > c.SPEC_ATT:
                i = c.CLASSIFIER_WILDCARD
                searching = False


def _unexpected_case(classifier: Classifier,
                     perception: list,
                     previous_perception: list) -> Classifier:

    classifier.q = classifier.q - c.BETA * classifier.q
    classifier.mark = previous_perception

    for i in range(len(perception)):
        if classifier.effect[i] != c.CLASSIFIER_WILDCARD:
            if (classifier.effect[i] != previous_perception[i] or
                    previous_perception[i] != perception[i]):
                return None

    child = Classifier.copy_from(classifier)

    for i in range(len(perception)):
        if (classifier.effect[i] == c.CLASSIFIER_WILDCARD and
                previous_perception[i] != perception[i]):
            child.condition[i] = previous_perception[i]
            child.effect[i] = perception[i]

    if classifier.q < 0.5:
        classifier.q = 0.5

    child.exp = 1

    return child


def _update_application_average(cla: Classifier, time: int):
    if cla.exp < 1 / c.BETA:
        cla.aav += (time - cla.tga - cla.aav) / cla.exp
    else:
        cla.aav += c.BETA * (time - cla.tga - cla.aav)

    # TGA? Should this be in ALP module?
    # Maybe naming convention should be changed
    cla.tga = time


def _add_alp_classifier(classifier: Classifier,
                        classifiers: list,
                        action_set: list) -> None:

    old_classifier = None

    for cla in action_set:
        if cla.is_subsumer(classifier):
            if old_classifier is None or cla.is_more_general(classifier):
                old_classifier = cla

    if old_classifier is None:
        for cla in action_set:
            if cla == classifier:
                old_classifier = cla

    if old_classifier is None:
        classifiers.append(classifier)
        action_set.append(classifier)
    else:
        old_classifier.q += c.BETA * (1 - old_classifier.q)


def _cover_triple(previous_perception: list,
                  perception: list,
                  action: int,
                  time: int) -> Classifier:

    child = Classifier()

    for i in range(len(perception)):
        if previous_perception[i] != perception[i]:
            child.condition[i] = previous_perception[i]
            child.effect[i] = perception[i]

    child.action = action
    child.alp = time
    child.tga = time
    child.t = time

    return child


def _does_anticipate_correctly(classifier: Classifier,
                               perception: list,
                               previous_perception: list) -> bool:
    """
    :param classifier: given classifier
    :param perception: current perception
    :param previous_perception: previous perception
    :return: True if classifier anticipates correctly, False otherwise
    """
    for i in range(c.CLASSIFIER_LENGTH):
        if classifier.effect == c.CLASSIFIER_WILDCARD:
            if previous_perception[i] != perception[i]:
                return False
        else:
            if (classifier.effect[i] != perception[i] or
                    previous_perception[i] == perception[i]):
                return False

    return True
