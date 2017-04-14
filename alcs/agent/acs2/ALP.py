import logging
from random import random
from copy import copy

from . import Classifier
from . import Constants as c
from .ACS2Utils import get_general_perception,\
    remove_classifier,\
    does_anticipate_correctly

logger = logging.getLogger(__name__)


def apply_alp(classifiers: list,
              action: int,
              time: int,
              action_set: list,
              perception: list,
              previous_perception: list,
              beta: float):

    was_expected_case = 0

    # We need to make sure that adding and removing
    # classifiers from and to action set is not reconsidered in the current
    # loop.
    old_action_set = copy(action_set)

    for cl in old_action_set:
        logger.info("[ALP] Running for %s", cl)
        cl.exp += 1
        _update_application_average(cl, time, beta)

        if does_anticipate_correctly(cl,
                                     perception,
                                     previous_perception):
            new_cl = _expected_case(cl, previous_perception, beta)
            was_expected_case = 1
        else:
            new_cl = _unexpected_case(cl,
                                      perception,
                                      previous_perception,
                                      beta)

            # If classifier's quality decreased below certain threshold
            # remove it.
            if cl.is_inadequate():
                logger.info("[ALP] Removing %s", cl)
                remove_classifier(classifiers, cl)
                remove_classifier(action_set, cl)

        if new_cl is not None:
            new_cl.t = time
            _add_alp_classifier(new_cl,
                                classifiers,
                                action_set,
                                beta)

    # If there wasn't any classifier in the action set that anticipated
    # new perception correctly generate one with proper cover triple.
    if was_expected_case == 0:
        logger.info("\tNo classifier in the action set that"
                    "anticipated correctly, generating using"
                    "the covering mechanism")
        new_cl = _cover_triple(previous_perception,
                               perception,
                               action,
                               time)
        _add_alp_classifier(new_cl, classifiers, action_set, beta)


def _expected_case(cl: Classifier,
                   perception: list,
                   beta: float,
                   u_max: int = None) -> Classifier:
    """
    Classifier predicted correctly (it's effect part was OK). We can generate
    a new classifier or not.

    No new classifier is generated when:
    - mark is empty or,
    - there is no difference between mark and perception
    In this case just increase it's quality.

    On the other hand when differences are detected between mark and
    perception - an offspring will be generated.

    :param cl:
    :param perception:
    :param beta:
    :param u_max:
    :return:
    """

    logger.info('\t\tExpected case occurred')

    if u_max is None:
        u_max = c.U_MAX

    diff = _get_differences(cl.mark, perception)

    if diff == get_general_perception():
        logger.info("\t\t\tIncreasing quality")
        cl.q += beta * (1 - cl.q)
        return None
    else:
        logger.info("\t\t\tGenerating new classifier")
        # Count number of non-# symbols in diff and condition part
        spec = cl.condition.number_of_specified_elements()
        spec_new = _number_of_spec(diff)

        child = Classifier.copy_from(cl)

        if spec == u_max:
            _remove_random_spec_element(child.condition)
            spec -= 1

            while (spec + spec_new) > u_max:
                if spec > 0 and random() < 0.5:
                    _remove_random_spec_element(child.condition)
                    spec -= 1
                else:
                    _remove_random_spec_element(diff)
                    spec_new -= 1
        else:
            while (spec + spec_new) > u_max:
                _remove_random_spec_element(diff)
                spec_new -= 1

        child.condition = diff

        if child.q < 0.5:
            child.q = 0.5

        # TODO: added by me
        cl.mark = Classifier.empty_mark()

        child.exp = 1

        logger.info("\t\t\tChild classifier: %s", child)
        return child


def _get_differences(mark: list, perception: list) -> list:
    """
    The difference determination needs to distinguish between two cases.
    1. Clear differences: are those where one or more attributes in the mark M
    do not contain the corresponding attribute in the perception.
    2. Fuzzy differences: are those where there is no clear difference but one
    or more attributes in the mark M specify more than the one value in
    perception.

    In the first case, one random clear difference is specified.
    In the latter case all differences are specified.

    :param mark: list of sets containing marking states
    :param perception: perception obtained by the agent

    :return: list of differences
    """
    diff = get_general_perception()

    if Classifier.is_marked(mark):
        type1 = 0  # counts when mark is different from perception
        type2 = 0  # counts when there are multiple marks in attribute

        for i, perceptron in enumerate(perception):
            if perceptron not in mark[i]:
                type1 += 1
            if len(mark[i]) > 1:
                type2 += 1

        if type1 > 0:
            # Clear differences - one or more absolute differences detected ->
            # specialize randomly chosen one
            type1 = int(random() * type1)
            for i, perceptron in enumerate(perception):
                if perceptron not in mark[i]:
                    if type1 == 0:
                        diff[i] = perceptron
                    type1 -= 1
        elif type2 > 0:
            for i, perceptron in enumerate(perception):
                if len(mark[i]) > 1:
                    diff[i] = perceptron

    return diff


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


def _unexpected_case(cl: Classifier,
                     perception: list,
                     previous_perception: list,
                     beta: float) -> Classifier:
    """
    Handles a situation when classifier does not predict correctly next state.

    A new classifier is generated only if the effect part of the to be
    investigated classifier cl can be modified to anticipate the change
    from previous perception to perception by only *specializing* attributes.

    If this is the case, a new classifier that is specialized in condition
    and effect where necessary is generated. It's experience is set to one.

    :param cl: classifier making predictions
    :param perception: current perception
    :param previous_perception: previous perception
    :param beta: learning rate
    :return: a new specialized classifier or None
    """

    logger.info('\t\tUnexpected case occurred')

    cl.q -= beta * cl.q
    cl.set_mark(previous_perception, perception)

    for i in range(len(perception)):
        if cl.effect[i] != c.CLASSIFIER_WILDCARD:
            if (cl.effect[i] != perception[i] or
                    previous_perception[i] == perception[i]):
                return None

    child = Classifier.copy_from(cl)

    for i in range(len(perception)):
        if (cl.effect[i] == c.CLASSIFIER_WILDCARD and
                previous_perception[i] != perception[i]):
            child.condition.specialize(i, previous_perception[i])
            child.effect[i] = perception[i]

    if cl.q < 0.5:
        cl.q = 0.5

    child.exp = 1

    # TODO: added be me
    child.mark = Classifier.empty_mark()

    logger.info("\t\t\tChild classifier: %s", child)
    return child


def _update_application_average(cl: Classifier, time: int, beta: float):
    """
    Procedure uses the moyenne adaptive modifee technique to reach
    an accurate value of the application average as soon as possible.
    Also the ALP timestamp is set in this procedure.

    :param cl: classifier to be updated
    :param time: current time
    :param beta: learning rate
    """
    if cl.exp < (1 / beta):
        cl.aav += (time - cl.t_alp - cl.aav) / cl.exp
    else:
        cl.aav += beta * (time - cl.t_alp - cl.aav)

    cl.t_alp = time


def _add_alp_classifier(cl: Classifier,
                        classifiers: list,
                        action_set: list,
                        beta: float) -> None:

    logger.info("\t\tTrying to insert %s", cl)

    old_cl = None

    for cla in action_set:
        if cla.can_subsume(cl):
            if old_cl is None or cla.is_more_general(cl):
                old_cl = cla

    if old_cl is None:
        for cla in action_set:
            if cla.condition == cl.condition and cla.effect == cl.effect:
                old_cl = cla

    if old_cl is None:
        logger.info("\t\t\tNo more general classifier found - adding: %s", cl)
        classifiers.append(cl)
        action_set.append(cl)
    else:
        logger.info("\t\t\tIncreasing existing classifiers quality: %s",
                    old_cl)
        old_cl.q += beta * (1 - old_cl.q)


def _cover_triple(previous_perception: list,
                  perception: list,
                  action: int,
                  time: int) -> Classifier:
    """
    Covering generates a classifier that specifies all changes from the
    previous to current perception in condition and effect part. The action
    part of the new classifier is set to the executed action.

    Triggered if a triple (situation-action-effect) is not represented by any
    classifier in the action set.

    :param previous_perception: previous perception
    :param perception: current perception
    :param action: new classifier action
    :param time: current step
    :return: new classifier covering desired triple
    """

    child = Classifier()
    child.action = action

    for i in range(len(perception)):
        if previous_perception[i] != perception[i]:
            child.condition.specialize(i, previous_perception[i])
            child.effect[i] = perception[i]

    child.exp = 0
    child.r = 0
    child.aav = 0
    child.t_alp = time
    child.t_ga = time
    child.t = time
    child.q = 0.5
    child.num = 1

    return child
