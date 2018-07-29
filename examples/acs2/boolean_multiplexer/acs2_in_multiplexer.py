import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer

from examples.acs2.boolean_multiplexer.utils import calculate_performance
from lcs.agents.acs2 import ACS2, Configuration


def _map_perception(perception):
    return [str(x) for x in perception]


if __name__ == '__main__':
    # Load desired environment
    mp = gym.make('boolean-multiplexer-6bit-v0')

    # Create agent
    cfg = Configuration(mp.env.observation_space.n, 2,
                        do_ga=False,
                        perception_mapper_fcn=_map_perception,
                        performance_fcn=calculate_performance,
                        performance_fcn_params={'ctrl_bits': 2})
    agent = ACS2(cfg)

    # Explore the environment
    population, _ = agent.explore(mp, 1500)

    # Exploit the environment
    agent = ACS2(cfg, population)
    population, metrics = agent.exploit(mp, 50)

    # See how it went
    for metric in metrics:
        print(metric)
