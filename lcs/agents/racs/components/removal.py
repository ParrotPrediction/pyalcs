import itertools
import logging
from typing import List

import numpy as np

from lcs.agents.racs import Classifier
from lcs.strategies.subsumption import does_subsume

logger = logging.getLogger(__name__)


def remove(k, population, action_set, match_set):
    cls = select_for_removal(population, k)

    if len(cls) == 0:
        return

    logger.info(f"Removing {len(cls)} classifiers")

    for lst in [x for x in [population, match_set, action_set] if x]:
        for cl in cls:
            lst.safe_remove(cl)


def select_for_removal(population, k=2) -> List[Classifier]:
    if len(population) < k:
        return []

    to_remove = []

    cls = np.random.choice(population, k, replace=False)
    theta_exp = cls[0].cfg.theta_exp

    for cl1, cl2 in itertools.permutations(cls, 2):
        if does_subsume(cl1, cl2, theta_exp):
            to_remove.append(cl2)

    return to_remove
