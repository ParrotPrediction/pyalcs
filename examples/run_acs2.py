import sys
import logging
from os.path import abspath, join, dirname

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from alcs.agent.acs2 import ACS2
from alcs.environment import maze

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=logging.INFO)


def plot_results(time, quality, numerosity):
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.plot(time, quality, 'r')
    ax1.set_title('Quality performance')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Quality')
    ax1.grid(True)

    ax2 = fig.add_subplot(212)
    ax2.plot(time, numerosity, 'g')
    ax2.set_title('Numerosity performance')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Numerosity')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # Load environment
    env = maze.Maze('mazes/m1.maze')

    # Initialize agent
    acs2 = ACS2(env)

    # Evaluate simulation
    time, quality, numerosity = acs2.evaluate(500)

    print("OK")
    plot_results(time, quality, numerosity)
