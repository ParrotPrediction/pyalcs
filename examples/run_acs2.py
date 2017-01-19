import sys
import logging
from os.path import abspath, join, dirname

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from acs.agent.acs2 import ACS2
from acs.environment import maze


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=logging.DEBUG)

if __name__ == '__main__':

    # Load environment
    env = maze.Maze('mazes/m1.maze')

    # Initialize agent
    acs2 = ACS2(env)

    # Evaluate simulation
    acs2.evaluate(5)
