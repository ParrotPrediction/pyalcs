import logging

import gym

# noinspection PyUnresolvedReferences
import gym_maze
from alcs import ACS2, ACS2Configuration
from integration.maze.utils import calculate_performance

# Configure logger
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    # Load desired environment
    maze = gym.make('BMaze4-v0')

    # Configure and create the agent
    cfg = ACS2Configuration(8, 8,
                            epsilon=1.0,
                            do_ga=False,
                            performance_fcn=calculate_performance)
    logging.info(cfg)

    # Explore the environment
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(maze, 50)

    # Exploit the environment
    agent = ACS2(cfg, population)
    population, exploit_metric = agent.exploit(maze, 10)

    for metric in exploit_metric:
        logging.info(metric)
