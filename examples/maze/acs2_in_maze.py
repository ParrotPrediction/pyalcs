import logging

import gym

# noinspection PyUnresolvedReferences
import gym_maze

from alcs import ACS2, ACS2Configuration
from examples.maze import calculate_knowledge

# Configure logger
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    # Load desired environment
    maze = gym.make('BMaze4-v0')

    # Configure and create the agent
    cfg = ACS2Configuration(8, 8, do_ga=True)
    logging.info(cfg)
    agent = ACS2(cfg)

    # Explore the environment
    logging.info("EXPLORE PHASE")
    population, metrics = agent.explore(maze, 50)

    for metric in metrics:
        logging.info(metric)

    logging.info("Knowledge obtained: {:.2f}%".format(
        calculate_knowledge(maze, population)))
