import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", default="Maze4-v0")
    parser.add_argument("--epsilon", default=1.0, type=float)
    parser.add_argument("--ga", action="store_true")
    parser.add_argument("--explore-trials", default=50, type=int)
    parser.add_argument("--exploit-trials", default=10, type=int)
    args = parser.parse_args()

    # Load desired environment
    maze = gym.make(args.environment)

    # Configure and create the agent
    cfg = Configuration(8, 8,
                        epsilon=args.epsilon,
                        do_ga=args.ga,
                        metrics_trial_frequency=1,
                        user_metrics_collector_fcn=maze_metrics)

    # Explore the environment
    logging.info("Exploring maze")
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(maze, args.explore_trials)

    for metric in explore_metrics:
        logger.info(metric)

    # Exploit the environment
    logging.info("Exploiting maze")
    agent = ACS2(cfg, population)
    population, exploit_metric = agent.exploit(maze, args.exploit_trials)

    for metric in exploit_metric:
        logger.info(metric)

    logger.info("Done")
