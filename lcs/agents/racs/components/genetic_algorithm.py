import numpy as np

from lcs.agents.racs import Classifier
from lcs.representations import UBR
from lcs.representations.RealValueEncoder import RealValueEncoder


def mutate(cl: Classifier, mu: float) -> None:
    """
    Tries to alternate the classifier condition and effect part.
    Each attribute (both lower/upper bound) have `mu` chances of being changed.

    Parameters
    ----------
    cl: Classifier
        classifier to be modified
    mu: float
        probability of executing mutation on single interval bound
    """
    encoder = cl.cfg.encoder
    noise_max = cl.cfg.mutation_noise

    for c, e in zip(cl.condition, cl.effect):
        if c != cl.cfg.classifier_wildcard:
            _mutate_attribute(c, encoder, noise_max, mu)
        if e != cl.cfg.classifier_wildcard:
            _mutate_attribute(e, encoder, noise_max, mu)


def _mutate_attribute(ubr: UBR, encoder: RealValueEncoder,
                      noise_max: float, mu: float):

    if np.random.random() < mu:
        noise = np.random.uniform(-noise_max, noise_max)
        x1p = encoder.decode(ubr.x1)
        ubr.x1 = encoder.encode(x1p, noise)

    if np.random.random() < mu:
        noise = np.random.uniform(-noise_max, noise_max)
        x2p = encoder.decode(ubr.x2)
        ubr.x2 = encoder.encode(x2p, noise)
