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
    rmpx = gym.make('real-multiplexer-3bit-v0')

    # Create agent
    encoder = RealValueEncoder(resolution_bits=2)
    cfg = Configuration(rmpx.observation_space.shape[0],
                        rmpx.action_space.n,
                        encoder=encoder,
                        epsilon=0.5,
                        do_ga=True,
                        theta_r=0.9,
                        theta_i=0.2,
                        theta_ga=100,
                        chi=0.5,
                        mu=0.15)

    agent = RACS(cfg)
    population, _ = agent.explore(rmpx, 100)
