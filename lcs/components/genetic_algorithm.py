from random import random, sample

from ..acs2 import ClassifiersList, Classifier


def roulette_wheel_parents_selection(pop: ClassifiersList, randomfunc=random):
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


def mutate(cl: Classifier, randomfunc=random):
    """
    Executes the generalizing mutation in the classifier.
    Specified attributes in classifier conditions are randomly
    generalized with `mu` probability.
    """
    for idx, cond in enumerate(cl.condition):
        if cond != cl.cfg.classifier_wildcard and \
                randomfunc() < cl.cfg.mu:
            cl.condition.generalize(idx)


def two_point_crossover(cl1: Classifier, cl2: Classifier, samplefunc=sample):
    """
    Executes two-point crossover using condition parts of two classifiers.
    :param cl1: first classifier
    :param cl2: second classifier
    :param samplefunc:
    """
    left, right = samplefunc(range(0, cl1.cfg.classifier_length + 1), 2)

    if left > right:
        left, right = right, left

    # Extract chromosomes from condition parts
    chromosome1 = cl1.condition[left:right]
    chromosome2 = cl2.condition[left:right]

    # Flip them
    for idx, el in enumerate(range(left, right)):
        cl1.condition[el] = chromosome2[idx]
        cl2.condition[el] = chromosome1[idx]
