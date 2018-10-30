import logging

import gym
# noinspection PyUnresolvedReferences
import gym_handeye

from examples.acs2.handeye.utils import handeye_metrics
from lcs.agents.acs2 import ACS2, Configuration

# Configure logger
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # Load desired environment
    hand_eye = gym.make('HandEye3-v0')

    # Configure and create the agent
    cfg = Configuration(hand_eye.observation_space.n, hand_eye.action_space.n,
                        epsilon=1.0,
                        do_ga=False,
                        do_action_planning=True,
                        action_planning_frequency=50,
                        user_metrics_collector_fcn=handeye_metrics)
    logging.info(cfg)

    # Explore the environment
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(hand_eye, 50)

    # Exploit the environment
    agent = ACS2(cfg, population)
    population, exploit_metric = agent.exploit(hand_eye, 10)

    for metric in exploit_metric:
        logging.info(metric)
