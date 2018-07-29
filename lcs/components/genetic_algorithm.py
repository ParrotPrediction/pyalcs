from random import random, sample
from typing import Callable

from ..acs2 import ACS2Classifier, ClassifiersList


def roulette_wheel_parents_selection(pop: ClassifiersList,
                                     randomfunc: Callable=random):
    """
    Select two classifiers from population according
    to roulette-wheel selection.
    """
    parent1, parent2 = None, None

    q_sum = sum(cl.q3num() for cl in pop)

    q_sel1 = randomfunc() * q_sum
    q_sel2 = randomfunc() * q_sum

    q_sel1, q_sel2 = sorted([q_sel2, q_sel1])

    q_counter = 0.0
    for cl in pop:
        q_counter += cl.q3num()
        if parent1 is None and q_counter > q_sel1:
            parent1 = cl
        if q_counter > q_sel2:
            parent2 = cl
            break

    return parent1, parent2


def mutate(cl: ACS2Classifier,
           mu: float,
           randomfunc: Callable=random):
    """
    Executes the generalizing mutation in the classifier.
    Specified attributes in classifier conditions are randomly
    generalized with `mu` probability.
    """
    for idx, cond in enumerate(cl.condition):
        if cond != cl.cfg.classifier_wildcard and randomfunc() < mu:
            cl.condition.generalize(idx)


def two_point_crossover(parent: ACS2Classifier,
                        donor: ACS2Classifier,
                        samplefunc: Callable=sample):
    """
    Executes two-point crossover using condition parts of two classifiers.
    :param parent: first classifier
    :param donor: second classifier
    :param samplefunc:
    """
    left, right = samplefunc(range(0, parent.cfg.classifier_length + 1), 2)

    if left > right:
        left, right = right, left

    # Extract chromosomes from condition parts
    chromosome1 = parent.condition[left:right]
    chromosome2 = donor.condition[left:right]

    # Flip them
    for idx, el in enumerate(range(left, right)):
        parent.condition[el] = chromosome2[idx]
        donor.condition[el] = chromosome1[idx]
