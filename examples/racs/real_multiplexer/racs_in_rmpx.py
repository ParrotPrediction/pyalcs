import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer

from lcs.agents.racs import Configuration, RACS

if __name__ == '__main__':
    # Load desired environment
    rmpx = gym.make('real-multiplexer-6bit-v0')

    # Create agent
    cfg = Configuration(rmpx.env.observation_space.shape[0], 2,
                        encoder_bits=3,
                        epsilon=1.0)
    agent = RACS(cfg)

    population, _ = agent.explore(rmpx, 10)
