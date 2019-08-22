import os, sys
import logging

sys.path.insert(0, os.path.abspath('../../../../openai-envs'))


import gym
# noinspection PyUnresolvedReferences
import gym_grid

from lcs.agents.acs2 import ACS2, Configuration

# Configure logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def print_cl(cl):
    action = None
    if cl.action == 0:
        action = '⬅'
    if cl.action == 1:
        action = '➡'
    if cl.action == 2:
        action = '⬆'
    if cl.action == 3:
        action = '⬇'
    print(f"{cl.condition} - {action} - {cl.effect} [fit: {cl.fitness:.3f}, r: {cl.r:.2f}, ir: {cl.ir:.2f}]")


if __name__ == '__main__':
    # Load desired environment
    grid = gym.make('grid-5-v0')

    # Configure and create the agent
    cfg = Configuration(
        classifier_length=2,
        number_of_possible_actions=4,
        epsilon=1.0,
        beta=0.3,
        gamma=0.8,
        theta_exp=50,
        theta_ga=50,
        do_ga=True,
        mu=0.02,
        u_max=2,
        metrics_trial_frequency=20)

    # Explore the environment
    logging.info("Exploring grid")
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(grid, 50)

    for cl in sorted(population, key=lambda c: -c.fitness):
        print_cl(cl)
