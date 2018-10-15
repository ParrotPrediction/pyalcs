import math
import random
from typing import Tuple

from scipy.stats import norm

from lcs.agents.racs import Classifier
from lcs.representations import UBR


def mutate(cl: Classifier, bounds: Tuple[int, int], mu: float) -> None:
    """
    Tries to generalize the classifier condition and effect part.
    Each attribute (both lower/upper bound) have `mu` chances of being broaden.

    Parameters
    ----------
    cl: Classifier
        classifier to be modified
    bounds: Tuple[int, int]
        tuple with minimum and maximum encoded value for the attribute
    mu: float
        probability of executing mutation on single bound
    """
    for idx, (c, e) in enumerate(zip(cl.condition, cl.effect)):
        if c != cl.cfg.classifier_wildcard:
            cl.condition[idx] = _mutate_attribute(c, bounds, mu)

        if e != cl.cfg.classifier_wildcard:
            cl.effect[idx] = _mutate_attribute(e, bounds, mu)


def _mutate_attribute(ubr: UBR, bounds: Tuple[int, int], mu: float) -> UBR:
    rmin, rmax = bounds[0], bounds[1]

    # Calculate global spread
    spread = _calculate_spread(rmax)

    lb, ub = ubr.lower_bound, ubr.upper_bound
    nlb, nub = lb, ub

    # Generate new lower bound
    if random.random() < mu:
        while True:
            nlb = _draw(lb, spread)
            if rmin <= nlb <= lb:
                break

    # Generate new upper bound
    if random.random() < mu:
        while True:
            nub = _draw(ub, spread)
            if ub <= nub <= rmax:
                break
    return UBR(nlb, nub)


def _calculate_spread(rmax: int) -> float:
    """
    Calculates the suggested spread for the neighbouring mutation points.
    For bigger ranges the spread should also be bigger.

    Parameters
    ----------
    rmax: int
        maximum value in the range

    Returns
    -------
    float
        spread value according to the range
    """
    return math.log(rmax)


def _draw(center: int, spread: float) -> int:
    """
    Draws a random number using Gaussian distribution and converts it
    into integer.

    Parameters
    ----------
    center: int
        center for Gaussian distribution generator
    spread: float
        spread for Gaussian distribution generator

    Returns
    -------
    int
        random number
    """
    return int(norm.rvs(center, spread))
