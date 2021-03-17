import numpy as np
from copy import copy

from lcs.agents.xcs import Configuration, Classifier, ClassifiersList


def run_ga(population: ClassifiersList,
           action_set: ClassifiersList,
           situation,
           time_stamp,
           cfg: Configuration):
    if action_set is None:
        return None

    temp_numerosity = sum(cl.numerosity for cl in action_set)
    if temp_numerosity == 0:
        return None

    if time_stamp - sum(cl.time_stamp * cl.numerosity for cl in action_set) \
            / temp_numerosity > cfg.ga_threshold:
        for cl in action_set:
            cl.time_stamp = time_stamp
        # select children
        parent1 = _select_offspring(action_set)
        parent2 = _select_offspring(action_set)
        child1 = copy(parent1)
        child2 = copy(parent2)
        child1.numerosity = 1
        child2.numerosity = 1
        child1.experience = 0
        child2.experience = 0
        # apply crossover
        if np.random.rand() < cfg.chi:
            _apply_crossover(child1, child2)
            child1.prediction = (parent1.prediction + parent2.prediction) / 2
            child1.error = 0.25 * (parent1.error + parent2.error) / 2
            child1.fitness = (parent1.fitness + parent2.fitness) / 2
            child2.prediction = child1.prediction
            child2.error = child1.error
            child2.fitness = child1.fitness
        # apply mutation on both children
        _apply_mutation(child1, cfg, situation)
        _apply_mutation(child2, cfg, situation)
        # apply subsumption or just insert into population
        if cfg.do_GA_subsumption:
            if parent1.does_subsume(child1):
                parent1.numerosity += 1
            elif parent2.does_subsume(child1):
                parent2.numerosity += 1
            else:
                population.insert_in_population(child1)

            if parent1.does_subsume(child2):
                parent1.numerosity += 1
            elif parent2.does_subsume(child2):
                parent2.numerosity += 1
            else:
                population.insert_in_population(child2)

        else:
            population.insert_in_population(child1)
            population.insert_in_population(child2)
        population.delete_from_population()


def _select_offspring(action_set: ClassifiersList) -> Classifier:
    fitness_sum = 0
    for cl in action_set:
        fitness_sum += cl.fitness
    choice_point = np.random.rand() * fitness_sum
    fitness_sum = 0
    for cl in action_set:
        fitness_sum += cl.fitness
        if fitness_sum > choice_point:
            return cl


def _apply_crossover(child1: Classifier, child2: Classifier):
    x = np.random.rand() * len(child1.condition)
    y = np.random.rand() * len(child1.condition)
    if x > y:
        x, y = y, x
    i = 0
    while i < y:
        if x <= i < y:
            child1.condition[i], child2.condition[i] =\
                child1.condition[i], child2.condition[i]
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
