from random import random, sample
from typing import Callable

from lcs.agents.acs2 import Classifier


def mutate(cl: Classifier,
           mu: float,
           randomfunc: Callable=random) -> None:
    """
    Executes the generalizing mutation in the classifier.
    Specified attributes in classifier conditions are randomly
    generalized with `mu` probability.
    """
    for idx, cond in enumerate(cl.condition):
        if cond != cl.cfg.classifier_wildcard and randomfunc() < mu:
            cl.condition.generalize(idx)


def two_point_crossover(parent: Classifier,
                        donor: Classifier,
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
