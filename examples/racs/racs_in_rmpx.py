import logging

import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer

from lcs.representations.RealValueEncoder import RealValueEncoder
from lcs.agents.racs import Configuration, RACS

# Configure logger
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    # Load desired environment
    rmpx = gym.make('real-multiplexer-6bit-v0')

    # Create agent
    encoder = RealValueEncoder(resolution_bits=7)
    cfg = Configuration(rmpx.observation_space.shape[0],
                        rmpx.action_space.n,
                        encoder=encoder,
                        epsilon=1.0,
                        do_ga=True,
                        theta_r=0.9,
                        theta_i=0.3,
                        theta_ga=100,
                        chi=0.5,
                        mu=0.15)

    agent = RACS(cfg)
    population, _ = agent.explore_exploit(rmpx, 3000)

    # filter reliable classifiers
    reliable = [cl for cl in population if cl.is_reliable()]
    reliable = sorted(reliable, key=lambda cl: -cl.fitness)

    for cl in reliable[:10]:
        print(cl)

    print("Done")
