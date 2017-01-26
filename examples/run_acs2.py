import sys
import logging
from os.path import abspath, join, dirname

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from alcs.agent.acs2 import ACS2
from alcs.environment import maze

from helpers.visualization import plot_performance


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=logging.WARN)


if __name__ == '__main__':

    # Load environment
    env = maze.Maze('mazes/MazeF1.maze')

    # Initialize agent
    acs2 = ACS2(env)

    # Evaluate simulation
    metrics = acs2.evaluate(500)

    # Plot results
    plot_performance(**metrics)
