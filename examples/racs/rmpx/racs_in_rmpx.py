import logging
from typing import Dict

import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer

from lcs.agents.racs import Configuration, RACS
from lcs.metrics import population_metrics
from lcs.representations.RealValueEncoder import RealValueEncoder

# Configure logger
from lcs.representations.utils import cover_ratio

logging.basicConfig(level=logging.INFO)

encoder = RealValueEncoder(resolution_bits=4)


def _rmpx_metrics(pop, env) -> Dict:
    metrics = {
        'fitness': (sum(cl.fitness for cl in pop) / len(pop)),
        'cover_ratio': (sum(cover_ratio(cl.condition, encoder) for cl in pop) /
                        len(pop))
    }

    # Add basic population metrics
    metrics.update(population_metrics(pop, env))

    return metrics


if __name__ == '__main__':
    # Load desired environment
    rmpx = gym.make('real-multiplexer-3bit-v0')

    # Create agent
    cfg = Configuration(rmpx.observation_space.shape[0],
                        rmpx.action_space.n,
                        encoder=encoder,
                        user_metrics_collector_fcn=_rmpx_metrics,
                        epsilon=1.0,
                        do_ga=True,
                        do_merging=True,
                        theta_r=0.9,
                        theta_i=0.3,
                        theta_ga=100,
                        chi=0.5,
                        mu=0.15)

    agent = RACS(cfg)
    population, metrics = agent.explore_exploit(rmpx, 2000)
    logging.info("Done")

    # print reliable classifiers
    logging.info("Reliable classifiers:")
    reliable = [cl for cl in population if cl.is_reliable()]
    reliable = sorted(reliable, key=lambda cl: -cl.fitness)

    for cl in reliable[:10]:
        logging.info(cl)

    logging.info("Population first 10")
    for cl in population[:10]:
        logging.info(cl)

    # print last metric
    logging.info(metrics[-1])
