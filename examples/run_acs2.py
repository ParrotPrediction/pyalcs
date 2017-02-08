import sys
import logging
from os.path import abspath, join, dirname

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

# ALCS
from alcs.agent.acs2 import ACS2
from alcs.environment.maze import Maze

# Helpers
from helpers.visualization import plot_performance


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=logging.WARN)


if __name__ == '__main__':

    # Load environment
    env = Maze('mazes/MazeF1.maze')

    # Initialize agent
    agent = ACS2(env)

    # Evaluate simulation
    classifiers, metrics = agent.evaluate(50)

    # Plot results
    plot_performance(**metrics)
