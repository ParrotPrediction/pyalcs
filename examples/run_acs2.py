import sys
import logging
from os.path import abspath, join, dirname

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from alcs.agent.acs2 import ACS2
from alcs.environment.maze import Maze


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=logging.DEBUG)


if __name__ == '__main__':

    # Load environment
    env = Maze('mazes/MazeF1.maze')

    # Initialize agent
    agent = ACS2(env)

    # Evaluate simulation
    classifiers, metrics = agent.evaluate(500)

    reliable = [c for c in classifiers if c.q > 0.9]

    print("Reliable classifiers: {}".format(len(reliable)))
