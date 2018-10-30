from typing import Dict

from lcs.metrics import population_metrics


def rmpx_metrics(pop, env) -> Dict:
    metrics = {
        'fitness': (sum(cl.fitness for cl in pop) / len(pop)),
        'cover_ratio': (sum(cl.condition.cover_ratio for cl in pop) / len(pop))
    }

    # Add basic population metrics
    metrics.update(population_metrics(pop, env))

    return metrics
