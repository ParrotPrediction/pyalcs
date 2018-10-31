import logging

import gym
# noinspection PyUnresolvedReferences
import gym_checkerboard

from lcs.agents.racs import Configuration, RACS
from lcs.agents.racs.metrics import count_averaged_regions
from lcs.metrics import population_metrics
from lcs.representations.RealValueEncoder import RealValueEncoder

# Configure logger
logging.basicConfig(level=logging.INFO)


def _checkerboard_metrics(population, environment):
    metrics = {
        'regions': count_averaged_regions(population)
    }

    # Add basic population metrics
    metrics.update(population_metrics(population, environment))

    return metrics


if __name__ == '__main__':
    # Load desired environment
    chckb = gym.make('checkerboard-2D-3div-v0')

    # Create agent
    encoder = RealValueEncoder(resolution_bits=4)
    cfg = Configuration(chckb.observation_space.shape[0],
                        chckb.action_space.n,
                        encoder=encoder,
                        user_metrics_collector_fcn=_checkerboard_metrics,
                        epsilon=0.5,
                        do_ga=True,
                        theta_r=0.9,
                        theta_i=0.2,
                        theta_ga=100,
                        chi=0.5,
                        mu=0.15)

    agent = RACS(cfg)
    population, metrics = agent.explore_exploit(chckb, 100)

    # print reliable classifiers
    reliable = [cl for cl in population if cl.is_reliable()]
    for cl in reliable:
        logging.info(cl)

    # print metrics
    for m in metrics:
        logging.info(m)
