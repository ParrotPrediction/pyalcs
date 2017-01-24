import sys
import logging
from os.path import abspath, join, dirname
from collections import Counter

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from alcs.agent.acs2 import ACS2
from alcs.environment import maze

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=logging.INFO)


def plot_results(time, cls_num, quality, found_reward, fitness, actions):
    fig = plt.figure()

    x = []
    for i, n in enumerate(cls_num):
        if found_reward[i]:
            x.append(n)
        else:
            x.append(None)

    ax1 = fig.add_subplot(221)
    ax1.plot(time, cls_num, 'b')
    ax1.plot(time, x, 'ro', linewidth=2.0)
    ax1.set_title('Total Classifiers')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Macro-classifiers')
    ax1.grid(True)

    y = []
    for i, n in enumerate(quality):
        if found_reward[i]:
            y.append(n)
        else:
            y.append(None)

    ax2 = fig.add_subplot(222)
    ax2.plot(time, quality, 'g')
    ax2.plot(time, y, 'ro', linewidth=2.0)
    ax2.set_title('Quality')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Quality per classifier')
    ax2.grid(True)

    z = []
    for i, n in enumerate(fitness):
        if found_reward[i]:
            z.append(n)
        else:
            z.append(None)

    ax3 = fig.add_subplot(223)
    ax3.plot(time, fitness, 'y')
    ax3.plot(time, z, 'ro', linewidth=2.0)
    ax3.set_title('Fitness')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Fitness per classifier')
    ax3.grid(True)

    a = Counter(actions)
    ax4 = fig.add_subplot(224)
    ax4.bar(a.keys(), a.values(), 0.8)
    ax4.set_title('Performed actions')
    ax4.set_xlabel('Action')
    ax4.set_ylabel('Occurrence')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # Load environment
    env = maze.Maze('mazes/m1.maze')

    # Initialize agent
    acs2 = ACS2(env)

    # Evaluate simulation
    time, cls_num, quality, found_reward, fitness, actions = acs2.evaluate(2500)

    # Plot results
    plot_results(time, cls_num, quality, found_reward, fitness, actions)
