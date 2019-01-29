import itertools
import logging
from typing import List

import numpy as np

from lcs import clip
from lcs.agents import PerceptionString
from lcs.agents.racs import Classifier, Condition, Effect
from lcs.representations import Interval

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
    noise_max = cl.cfg.mutation_noise
    wildcard = cl.cfg.classifier_wildcard

    for c, e in zip(cl.condition, cl.effect):
        if c != wildcard and e != wildcard:
            _widen_attribute(c, noise_max, mu)
            _widen_attribute(e, noise_max, mu)


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


def _widen_attribute(interval: Interval, noise_max: float, mu: float):

    # TODO: we should modify both condition and effect parts with the
    # same noise.
    # This mutation can also create non suitable classifier!
    if np.random.random() < mu:
        noise = np.random.uniform(-noise_max, noise_max)
        interval.x1 = clip(interval.x1 + noise)

    if np.random.random() < mu:
        noise = np.random.uniform(-noise_max, noise_max)
        interval.x2 = clip(interval.x2 + noise)


def _flatten(ps: PerceptionString) -> List:
    """
    Flattens the perception string interval predicate into a flat list

    Returns
    -------
    List
        list of all alleles (encoded)
    """
    return list(itertools.chain.from_iterable(
        map(lambda ip: (ip.x1, ip.x2), ps)))


def _unflatten(flatten: List) -> List[Interval]:
    """
    Unflattens list by creating pairs of Intervals using consecutive list items

    Parameters
    ----------
    flatten: List[float]
        Flat list of perceptions

    Returns
    -------
    List[Interval]
        List of created Intervals
    """
    # Make sure we are not left with any outliers
    assert len(flatten) % 2 == 0
    return [Interval(flatten[i], flatten[i + 1])
            for i in range(0, len(flatten), 2)]
