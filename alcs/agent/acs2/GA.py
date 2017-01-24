import logging
from random import random

from . import Classifier
from . import Constants as c
from .ACS2Utils import get_general_perception, generate_random_int_number,\
    remove_classifier

logger = logging.getLogger(__name__)

# Genetic Generalization Mechanism


def apply_ga(classifiers: list,
             action_set: list,
             time: int,
             theta_ga: int = None,
             x=None):

    if theta_ga is None:
        theta_ga = c.THETA_GA

    if x is None:
        x = c.X

    if _should_fire(action_set, time, theta_ga):
        logger.debug("Applying GA module")

        for cl in action_set:
            cl.t_ga = time

            parent1 = _select_offspring(action_set)
            parent2 = _select_offspring(action_set)

            child1 = Classifier.copy_from(parent1)
            child2 = Classifier.copy_from(parent2)

            child1.num = 1
            child2.num = 1

            child1.exp = 1
            child2.exp = 1

            _apply_ga_mutation(child1)
            _apply_ga_mutation(child2)

            if random() < x:
                _apply_crossover(child1, child2)

                child1.r = (parent1.r + parent2.r) / 2
                child2.r = (parent1.r + parent2.r) / 2

                child1.q = (parent1.q + parent2.q) / 2
                child2.q = (parent1.q + parent2.q) / 2

            child1.q /= 2
            child2.q /= 2

            _delete_classifiers(classifiers, action_set)

            if child1.condition != get_general_perception():
                _add_ga_classifier(classifiers, action_set, child1)

            if child2.condition != get_general_perception():
                _add_ga_classifier(classifiers, action_set, child2)


def _should_fire(action_set: list, time: int, theta_ga: int) -> bool:
    """
    Check if GA should take place by examining t_ga timestamps with current
    time.

    :param action_set: list of classifiers (action_set)
    :param time: current time
    :param theta_ga: GA application threshold
    :return: True if GA will take place, false otherwise
    """

    overall_tga = sum(cl.t_ga * cl.num for cl in action_set)
    overall_num = sum(cl.num for cl in action_set)

    return (time - overall_tga) / overall_num > theta_ga


def _select_offspring(action_set: list) -> Classifier:
    """
    The process chooses a classifier fro reproduction in action set
    proportional to its quality to the power of three. First, the sum of all
    values in set is computed. Next, the roulette-wheel spun. Finally, the
    classifier is chosen according to the roulette-wheel result.

    :param action_set: set of classifiers
    :return: selected classifier
    """
    quality_sum = sum(cl.q ** 3 for cl in action_set)
    choice_point = random() * quality_sum

    quality_sum = 0

    for cl in action_set:
        quality_sum += cl.q ** 3
        if quality_sum > choice_point:
            return cl


def _apply_ga_mutation(cl: Classifier, mu: float = None) -> None:
    """
    Looks for classifier condition elements (not generic), and tries to
    generify each one with probability mu

    :param cl: classifier to apply mutation on
    :param mu: mutation rate
    """
    if mu is None:
        mu = c.MU

    for i in range(len(cl.condition)):
        if cl.condition[i] != c.CLASSIFIER_WILDCARD:
            if random() < mu:
                cl.condition[i] = c.CLASSIFIER_WILDCARD


def _add_ga_classifier(classifiers: list,
                       action_set: list,
                       cl: Classifier) -> None:

    old_cl = None

    for c in action_set:
        if c.is_subsumer(cl):
            if old_cl is None or c.is_more_general(old_cl):
                old_cl = c

    if old_cl is None:
        for c in action_set:
            if c.condition == cl.condition and c.effect == cl.effect:
                old_cl = c

    if old_cl is None:
        classifiers.append(cl)
        action_set.append(cl)
    else:
        if not Classifier.is_marked(old_cl.mark):
            old_cl.num += 1


def _apply_crossover(cl1: Classifier, cl2: Classifier):
    if cl1.effect != cl2.effect:
        return

    # Break points
    x = generate_random_int_number(len(cl1.condition))
    y = None

    while True:
        y = generate_random_int_number(len(cl1.condition))
        if x != y:
            break

    if x > y:
        tmp = x
        x = y
        y = tmp

    i = 0

    while True:
        if x <= i < y:
            tp = cl1.condition[i]
            cl1.condition[i] = cl2.condition[i]
            cl2.condition[i] = tp

        i += 1
        if i > y:
            break


def _delete_classifiers(classifiers: list,
                        action_set: list,
                        in_size: int = None,
                        theta_as: int = None):

    if in_size is None:
        in_size = c.IN_SIZE

    if theta_as is None:
        theta_as = c.THETA_AS

    action_set_numerosity = sum(cl.num for cl in action_set)

    while in_size + action_set_numerosity > theta_as:
        cl_del = None

        for cl in classifiers:
            if random() < (1 / 3):
                if cl_del is None:
                    cl_del = cl
                else:
                    if cl.q - cl_del.q < -0.1:
                        cl_del = cl
                    if abs(cl.q - cl_del.q) <= 0.1:
                        if (Classifier.is_marked(cl) and
                                not Classifier.is_marked(cl_del.mark)):
                            cl_del = cl
                        elif (Classifier.is_marked(cl) or
                                not Classifier.is_marked(cl_del.mark)):
                            if cl.aav > cl_del.aav:
                                cl_del = cl

        if cl_del is not None:
            if cl_del.num > 1:
                cl_del.num -= 1
            else:
                remove_classifier(classifiers, cl)
                remove_classifier(action_set, cl)

        summation = 0

        for classifier in action_set:
            summation += classifier.num
