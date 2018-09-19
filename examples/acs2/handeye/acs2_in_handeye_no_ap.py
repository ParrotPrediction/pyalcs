import logging

import gym

# noinspection PyUnresolvedReferences
import gym_handeye
from lcs.agents.acs2 import ACS2, Configuration
from examples.acs2.handeye.utils import calculate_performance

# Configure logger
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # Load desired environment
    hand_eye = gym.make('HandEye3-v0')

    # Configure and create the agent
    cfg = Configuration(hand_eye.observation_space.n, hand_eye.action_space.n,
                            epsilon=1.0,
                            do_ga=False,
                            do_action_planning=False,
                            performance_fcn=calculate_performance)
    logging.info(cfg)

    # Explore the environment
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(hand_eye, 50)

    # Exploit the environment
    agent = ACS2(cfg, population)
    population, exploit_metric = agent.exploit(hand_eye, 10)

    for metric in exploit_metric:
        logging.info(metric)
