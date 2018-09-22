import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer

from lcs.agents.racs import Configuration, RACS

if __name__ == '__main__':
    # Load desired environment
    rmpx = gym.make('real-multiplexer-6bit-v0')

    # Create agent
    cfg = Configuration(rmpx.env.observation_space.shape[0], 2,
                        encoder_bits=2,
                        epsilon=0.5,
                        do_ga=True,
                        theta_r=0.9,
                        theta_i=0.2,
                        theta_ga=100,
                        chi=0.5,
                        mu=0.15)
    agent = RACS(cfg)
    # favour most general condition
    # and least general effect

    population, _ = agent.explore(rmpx, 100_000)