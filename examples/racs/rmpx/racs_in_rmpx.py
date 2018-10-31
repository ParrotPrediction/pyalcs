import logging
from typing import Dict

import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer

from lcs.agents.racs import Configuration, RACS
from lcs.metrics import population_metrics
from lcs.representations.RealValueEncoder import RealValueEncoder

# Configure logger
logging.basicConfig(level=logging.INFO)


def _rmpx_metrics(pop, env) -> Dict:
    metrics = {
        'fitness': (sum(cl.fitness for cl in pop) / len(pop)),
        'cover_ratio': (sum(cl.condition.cover_ratio for cl in pop) / len(pop))
    }

    # Add basic population metrics
    metrics.update(population_metrics(pop, env))

    return metrics


if __name__ == '__main__':
    # Load desired environment
    rmpx = gym.make('real-multiplexer-3bit-v0')

    # Create agent
    encoder = RealValueEncoder(resolution_bits=5)
    cfg = Configuration(rmpx.observation_space.shape[0],
                        rmpx.action_space.n,
                        encoder=encoder,
                        user_metrics_collector_fcn=_rmpx_metrics,
                        epsilon=1.0,
                        do_ga=True,
                        theta_r=0.9,
                        theta_i=0.3,
                        theta_ga=100,
                        chi=0.5,
                        mu=0.15)

    agent = RACS(cfg)
    population, metrics = agent.explore_exploit(rmpx, 100)
    logging.info("Done")

    # print reliable classifiers
    reliable = [cl for cl in population if cl.is_reliable()]
    reliable = sorted(reliable, key=lambda cl: -cl.fitness)

    for cl in reliable[:10]:
        logging.info(cl)

    # print metrics
    for m in metrics:
        logging.info(m)
