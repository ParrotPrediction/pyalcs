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
