import itertools
import logging
import math
from typing import List

import numpy as np

from lcs.agents import PerceptionString
from lcs.agents.racs import Classifier, Condition, Effect
from lcs.representations import UBR
from lcs.representations.utils import add_from_both_sides
from lcs.representations.RealValueEncoder import RealValueEncoder

logger = logging.getLogger(__name__)


def mutate(cl: Classifier, mu: float) -> None:
    """
    Tries to alternate (widen) the classifier condition and effect part.
    Each attribute (both lower/upper bound) have `mu` chances of being changed.

    Parameters
    ----------
    cl: Classifier
        classifier to be modified
    mu: float
        probability of executing mutation on single interval bound
    """
    encoder = cl.cfg.encoder
    wildcard = cl.cfg.classifier_wildcard

    for c, e in zip(cl.condition, cl.effect):
        if c != wildcard and e != wildcard:
            _widen(c, e, encoder, mu)


def crossover(parent: Classifier, donor: Classifier):
    assert parent.cfg.classifier_length == donor.cfg.classifier_length

    # flatten parent and donor perception strings
    p_cond_flat = _flatten(parent.condition)
    d_cond_flat = _flatten(donor.condition)
    p_effect_flat = _flatten(parent.effect)
    d_effect_flat = _flatten(donor.effect)

    # select crossing points
    left, right = sorted(np.random.choice(
        range(0, len(p_cond_flat) + 1), 2, replace=False))

    assert left < right

    # extract chromosomes
    p_cond_chromosome = p_cond_flat[left:right]
    d_cond_chromosome = d_cond_flat[left:right]
    p_effect_chromosome = p_effect_flat[left:right]
    d_effect_chromosome = d_effect_flat[left:right]

    # Flip everything
    p_cond_flat[left:right] = d_cond_chromosome
    d_cond_flat[left:right] = p_cond_chromosome
    p_effect_flat[left:right] = d_effect_chromosome
    d_effect_flat[left:right] = p_effect_chromosome

    # Rebuild proper perception strings
    parent.condition = Condition(_unflatten(p_cond_flat), cfg=parent.cfg)
    donor.condition = Condition(_unflatten(d_cond_flat), cfg=donor.cfg)
    parent.effect = Effect(_unflatten(p_effect_flat), cfg=parent.cfg)
    donor.effect = Effect(_unflatten(d_effect_flat), cfg=parent.cfg)


def _widen(c_ubr: UBR, e_ubr: UBR, encoder: RealValueEncoder, mu: float):
    """
    This method add the same noise to condition and effect interval.
    Noise is added in genotype space (encoded) to both sides of the interval

    Parameters
    ----------
    c_ubr: UBR
        condition allele
    e_ubr: UBR
        effect allele
    encoder: RealValueEncoder
    mu: float
    """
    if np.random.random() < mu:
        # max noise is 10% of available range
        max_noise = math.ceil(encoder.range[1] / 10)

        # divide generated noise by half to add to each interval side
        noise = math.ceil(np.random.randint(0, max_noise + 1) / 2)

        (lb, ub) = encoder.range

        add_from_both_sides(c_ubr, noise, lb, ub)
        add_from_both_sides(e_ubr, noise, lb, ub)


def _flatten(ps: PerceptionString) -> List[int]:
    """
    Flattens the perception string interval predicate into a flat list

    Returns
    -------
    List
        list of all alleles (encoded)
    """
    return list(itertools.chain.from_iterable(
        map(lambda ip: (ip.x1, ip.x2), ps)))


def _unflatten(flatten: List[int]) -> List[UBR]:
    """
    Unflattens list by creating pairs of UBR using consecutive list items

    Parameters
    ----------
    flatten: List[int]
        Flat list of encoded perceptions

    Returns
    -------
    List[UBR]
        List of created UBRs
    """
    # Make sure we are not left with any outliers
    assert len(flatten) % 2 == 0
    return [UBR(flatten[i], flatten[i + 1]) for i in range(0, len(flatten), 2)]
