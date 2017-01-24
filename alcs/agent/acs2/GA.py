import logging
from random import random

from . import Classifier
from . import Constants as c
from .ACS2Utils import remove

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

        for classifier in action_set:
            classifier.t_ga = time

            parent1 = _select_offspring(action_set)
            parent2 = _select_offspring(action_set)

            child1 = Classifier.copy_from(parent1)
            child2 = Classifier.copy_from(parent2)

            child1.num += 1
            child2.num += 1

            child1.exp += 1
            child2.exp += 1

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

            # _delete_classifier(classifiers, action_set)

            if child1.condition != \
                    [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH:
                _add_ga_classifier(classifiers, action_set, child1)

            if child2.condition != \
                    [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH:
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
    quality_sum = 0

    for classifier in action_set:
        quality_sum += classifier.q ** 3

    # quality_sum = sum(cls.q ** 3 for cls in action_set)
    choice_point = random() * quality_sum
    partial_quality_sum = 0

    for classifier in action_set:
        partial_quality_sum += classifier.q ** 3
        if partial_quality_sum > choice_point:
            return classifier


def _apply_ga_mutation(classifier: Classifier, mu=None) -> None:
    """
    Looks for classifier condition elements (not generic), and tries to
    generify each one with probability mu

    :param classifier: classifier to apply mutation on
    :param mu: probability
    """
    if mu is None:
        mu = c.MU

    for i in range(len(classifier.condition)):
        if classifier.condition[i] != c.CLASSIFIER_WILDCARD:
            if random() < mu:
                classifier.condition[i] = c.CLASSIFIER_WILDCARD


def _add_ga_classifier(classifiers: list,
                       action_set: list,
                       classifier: Classifier) -> None:

    old_cls = None

    for cls in action_set:
        if cls.is_subsumer(classifier):
            if old_cls is None or old_cls.is_more_general(classifier):
                old_cls = cls

    if old_cls is None:
        for cls in action_set:
            if cls == classifier:
                old_cls = cls

    if old_cls is None:
        classifiers.append(classifier)
        action_set.append(classifier)
    else:
        if old_cls.mark is None:
            old_cls.num += 1


def _apply_crossover(cl1: Classifier, cl2: Classifier):
    if cl1.effect != cl2.effect:
        return

    x = random() * (len(cl1.condition) + 1)
    while True:
        y = random() * (len(cl1.condition) + 1)
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
        if i <= y:
            break


def _delete_classifier(classifiers: list, action_set: list):
    summation = 0
    for cl in action_set:
        summation += cl.num

    while c.IN_SIZE + summation > c.THETA_AS:
        cl_del = None
        for cl in classifiers:
            if random() < 1 / 3:
                if cl_del is None:
                    cl_del = cl
                else:
                    if cl.q - cl_del.q < -0.1:
                        cl_del = classifier
                    if abs(cl.q - cl_del.q) <= 0.1:
                        if __name__ == '__main__':
                            if cl.mark is not None and cl_del.mark is None:
                                cl_del = classifier
                            elif cl.mark is not None or cl_del.mark is None:
                                if cl.aav > cl_del.aav:
                                    cl_del = cl

        if cl_del is not None:
            if cl_del.num > 1:
                cl_del.num -= 1
            else:
                classifiers.remove(classifier)  # TODO: nie ma tej zmiennej
                remove(classifier, action_set)

        summation = 0

        for classifier in action_set:
            summation += classifier.num
