import logging
import pickle

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
    cfg = ACS2Configuration(8, 8, epsilon=1.0, do_ga=False)
    logging.info(cfg)
    agent = ACS2(cfg)

    # Explore the environment
    logging.info("EXPLORE PHASE")
    population, metrics = agent.explore_exploit(maze, 50)

    for metric in metrics:
        logging.info(metric)

    logging.info("Knowledge obtained: {:.2f}%".format(
        calculate_knowledge(maze, population)))

    # Store metrics in file
    logging.info("Dumping data to files ...")
    pickle.dump(population, open("maze_population.pkl", "wb"))
    pickle.dump(metrics, open("maze_metrics.pkl", "wb"))
