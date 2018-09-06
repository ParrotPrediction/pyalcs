import random
from typing import Callable, Dict


def roulette_wheel_selection(population, fitnessfunc: Callable):
    """
    Select two objects from population according
    to roulette-wheel selection.

    Parameters
    ----------
    population
        population of objects (probably classifiers)
    fitnessfunc: Callable
        function evaluating fitness for each classifier. Very often cl.q^3

    Returns
    -------
    tuple
        two classifiers selected as parents
    """
    choices = {cl: fitnessfunc(cl) for cl in population}

    parent1 = _weighted_random_choice(choices)
    parent2 = _weighted_random_choice(choices)

    return parent1, parent2


def _weighted_random_choice(choices: Dict):
    max = sum(choices.values())
    pick = random.uniform(0, max)
    current = 0

    for key, value in choices.items():
        current += value
        if current > pick:
            return key
