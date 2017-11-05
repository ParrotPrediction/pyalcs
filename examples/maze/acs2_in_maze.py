import logging

from alcs import ACS2, ACS2Configuration

import gym
import gym_maze


# Configure logger
logging.basicConfig(level=logging.INFO)


def _calculate_knowledge(maze, population):
    """
    Analyzes all possible transition in maze environment and checks if there
    is a reliable classifier for it.
    :param maze: maze object
    :param population: list of classifiers
    :return: percentage of knowledge
    """
    transitions = maze.env.get_all_possible_transitions()

    # Take into consideration only reliable classifiers
    reliable_classifiers = [c for c in population if c.is_reliable()]

    # Count how many transitions are anticipated correctly
    nr_correct = 0

    # For all possible destinations from each path cell
    for start, action, end in transitions:

        p0 = maze.env.maze.perception(*start)
        p1 = maze.env.maze.perception(*end)

        if any([True for cl in reliable_classifiers
                if cl.predicts_successfully(p0, action, p1)]):
            nr_correct += 1

    return nr_correct / len(transitions) * 100.0

if __name__ == '__main__':

    # Load desired environment
    maze = gym.make('BMaze4-v0')

    # Configure and create the agent
    cfg = ACS2Configuration(8, 8, do_ga=True)
    logging.info(cfg)
    agent = ACS2(cfg)

    # Explore the environment
    logging.info("EXPLORE PHASE")
    population, metrics = agent.explore(maze, 5)

    for metric in metrics:
        logging.info(metric)

    logging.info("Knowledge obtained: {:.2f}%".format(
        _calculate_knowledge(maze, population)))
