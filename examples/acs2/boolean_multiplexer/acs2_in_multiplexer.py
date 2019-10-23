import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer

from examples.acs2.boolean_multiplexer.utils import reliable_cl_exists
from lcs.agents import EnvironmentAdapter
from lcs.agents.acs2 import ACS2, Configuration


def mpx_metrics(pop, env):
    return {
        'population': len(pop),
        'reliable_cl_exists': reliable_cl_exists(env, pop, ctrl_bits=2)
    }


class MultiplexerAdapter(EnvironmentAdapter):
    @classmethod
    def to_genotype(cls, env_state):
        return [str(x) for x in env_state]


if __name__ == '__main__':
    # Load desired environment
    mp = gym.make('boolean-multiplexer-6bit-v0')

    # Create agent
    cfg = Configuration(mp.env.observation_space.n, 2,
                        do_ga=False,
                        environment_adapter=MultiplexerAdapter(),
                        metrics_trial_frequency=50,
                        user_metrics_collector_fcn=mpx_metrics)
    agent = ACS2(cfg)

    # Explore the environment
    population, explore_metrics = agent.explore(mp, 1500)

    # Exploit the environment
    agent = ACS2(cfg, population)
    population, exploit_metrics = agent.exploit(mp, 50)

    # See how it went
    for metric in explore_metrics:
        print(metric)
