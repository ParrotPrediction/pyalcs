import gym

# noinspection PyUnresolvedReferences
import gym_multiplexer
from alcs import ACS2, ACS2Configuration
from integration.multiplexer.utils import calculate_performance

if __name__ == '__main__':
    # Load desired environment
    mp = gym.make('boolean-multiplexer-6bit-v0')

    # Create agent
    cfg = ACS2Configuration(mp.env.observation_space.n, 2,
                            do_ga=False,
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
