import random

import numpy as np

from lcs.agents.xcs import Configuration, Classifier, ClassifiersList


def run_ga(population: ClassifiersList,
           action_set: ClassifiersList,
           situation,
           time_stamp,
           cfg: Configuration):

    if action_set is None:
        return

    assert isinstance(population, ClassifiersList)
    assert isinstance(action_set, ClassifiersList)
    assert isinstance(cfg, Configuration)

    if time_stamp - (sum(cl.time_stamp * cl.numerosity for cl in action_set)
                     / (sum(cl.numerosity for cl in action_set) or 1)) > cfg.ga_threshold:
        for cl in action_set:
            cl.time_stamp = time_stamp

        parent1 = _select_offspring(action_set)
        parent2 = _select_offspring(action_set)

        child1, child2 = _make_children(parent1, parent2, cfg, time_stamp)

        if np.random.rand() < cfg.chi:
            _apply_crossover(child1, child2, parent1, parent2)

        _apply_mutation(child1, cfg, situation)
        _apply_mutation(child2, cfg, situation)

        _perform_insertion_or_subsumption(cfg, population,
                                          child1, child2,
                                          parent1, parent2)


def _perform_insertion_or_subsumption(cfg: Configuration, population: ClassifiersList,
                                      child1: Classifier, child2: Classifier,
                                      parent1: Classifier, parent2: Classifier):

    assert isinstance(child1, Classifier)
    assert isinstance(child2, Classifier)
    assert isinstance(parent1, Classifier)
    assert isinstance(parent2, Classifier)

    if cfg.do_GA_subsumption:
        for child in child1, child2:
            if parent1.does_subsume(child):
                parent1.numerosity += 1
            elif parent2.does_subsume(child):
                parent2.numerosity += 1
            else:
                population.insert_in_population(child)
            population.delete_from_population()
    else:
        for child in child1, child2:
            population.insert_in_population(child)
            population.delete_from_population()


def _make_children(parent1, parent2, cfg, time_stamp):
    assert isinstance(parent1, Classifier)
    assert isinstance(parent2, Classifier)

    child1 = Classifier(cfg, parent1.condition, parent1.action, time_stamp)
    child2 = Classifier(cfg, parent2.condition, parent2.action, time_stamp)

    return child1, child2


def _select_offspring(action_set: ClassifiersList) -> Classifier:

    assert isinstance(action_set, ClassifiersList)

    # TODO: insert generator to calculate fitness_sum
    fitness_sum = 0
    for cl in action_set:
        fitness_sum += cl.fitness
    choice_point = np.random.rand() * fitness_sum
    fitness_sum = 0
    for cl in action_set:
        fitness_sum += cl.fitness
        if fitness_sum > choice_point:
            return cl
    return action_set[random.randrange(len(action_set))]


def _apply_crossover(child1: Classifier, child2: Classifier,
                     parent1: Classifier, parent2: Classifier):

    _apply_crossover_in_area(child1,
                             child2,
                             np.random.rand() * len(child1.condition),
                             np.random.rand() * len(child1.condition)
                             )

    for child in child1, child2:
        child.prediction = (parent1.prediction + parent2.prediction) / 2
        child.error = 0.25 * (parent1.error + parent2.error) / 2
        child.fitness = 0.1 * (parent1.fitness + parent2.fitness) / 2


def _apply_crossover_in_area(child1: Classifier, child2: Classifier, x, y):
    if x > y:
        x, y = y, x
    if x > len(child2.condition):
        return
    if y > len(child2.condition):
        y = len(child2.condition)
    i = 0
    while i < y:
        if x <= i < y:
            temp = child1.condition[i]
            child1.condition[i] = child2.condition[i]
            child2.condition[i] = temp
        i += 1


def _apply_mutation(child: Classifier,
                    cfg: Configuration,
                    situation):
    i = 0
    while i < len(child.condition):
        if np.random.rand() < cfg.mutation_chance:
            if child.condition[i] == child.condition.WILDCARD:
                child.condition[i] = situation[i]
            else:
                child.condition[i] = child.condition.WILDCARD
        i += 1
    if np.random.rand() < cfg.mutation_chance:
        child.action = np.random.randint(cfg.number_of_actions)
