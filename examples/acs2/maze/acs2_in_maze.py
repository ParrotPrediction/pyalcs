import logging

import gym
# noinspection PyUnresolvedReferences
import gym_maze

from examples.acs2.maze.utils import maze_knowledge
from lcs.agents.acs2 import ACS2, Configuration

# Configure logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def maze_metrics(population, environment):
    return {
        'population': len(population),
        'knowledge': maze_knowledge(population, environment)
    }


if __name__ == '__main__':

    # Load desired environment
    maze = gym.make('BMaze4-v0')

    # Configure and create the agent
    cfg = Configuration(8, 8,
                        epsilon=1.0,
                        do_ga=False,
                        user_metrics_collector_fcn=maze_metrics)

    # Explore the environment
    logging.info("Exploring maze")
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(maze, 50)

    for metric in explore_metrics:
        logger.info(metric)

    # Exploit the environment
    logging.info("Exploiting maze")
    agent = ACS2(cfg, population)
    population, exploit_metric = agent.exploit(maze, 10)

    for metric in exploit_metric:
        logger.info(metric)

