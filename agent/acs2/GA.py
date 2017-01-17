from agent.acs2 import Constants as c
from agent.acs2.Classifier import Classifier

from random import random


def apply_ga(classifiers: list,
             action_set: list,
             time: int,
             theta_ga=None,
             x=None):

    if theta_ga is None:
        theta_ga = c.THETA_GA

    if x is None:
        x = c.X

    sumNum = 0
    sumTgaN = 0

    for classifier in action_set:
        sumNum += classifier.num
        sumTgaN += classifier.tga * classifier.num

    if (time - sumTgaN) / sumNum * theta_ga:
        for classifier in action_set:
            classifier.tga = time

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

            if random < x:
                _apply_crossover(child1, child2)

                child1.r = (parent1.r + parent2.r) / 2
                child2.r = (parent1.r + parent2.r) / 2

                child1.q = (parent1.q + parent2.q) / 2
                child2.q = (parent1.q + parent2.q) / 2

            child1.q /= 2
            child2.q /= 2

            _delete_classifier(classifiers, action_set)

            if child1.condition != \
                    [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH:
                _add_ga_classifier(classifiers, action_set, child1)

            if child2.condition != \
                    [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH:
                _add_ga_classifier(classifiers, action_set, child2)


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
            # TODO: check
            if cls.equals(classifier):
                old_cls = cls

    if old_cls is None:
        classifiers.append(classifier)
        action_set.append(classifier)
    else:
        if old_cls.mark is None:
            old_cls.num += 1
