import logging

import gym

import sys
sys.path.append('/home/e-dzia/openai-envs')

# noinspection PyUnresolvedReferences
import gym_handeye
from alcs import ACS2, ACS2Configuration
from integration.handeye.utils import calculate_performance

# Configure logger
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    # Load desired environment
    hand_eye = gym.make('HandEye3-v0')

    # Configure and create the agent
    cfg = ACS2Configuration(10, 6,
                            epsilon=1.0,
                            do_ga=False,
                            do_action_planning=True,
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
