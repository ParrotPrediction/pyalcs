import logging
import os
import sys

from lcs.agents.acs2 import ACS2, Configuration


sys.path.insert(0, os.path.abspath('../../../../openai-envs'))

import gym  # noqa: E402
# noinspection PyUnresolvedReferences
import gym_grid  # noqa: E402


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
    print(f"{cl.condition} - {action} - {cl.effect} "
          f"[fit: {cl.fitness:.3f}, r: {cl.r:.2f}, ir: {cl.ir:.2f}]")


if __name__ == '__main__':
    # Load desired environment
    grid = gym.make('grid-10-v0')

    # Configure and create the agent
    cfg = Configuration(
        classifier_length=2,
        number_of_possible_actions=4,
        epsilon=0.9,
        beta=0.03,
        gamma=0.97,
        theta_i=0.1,
        theta_as=10,
        theta_exp=50,
        theta_ga=50,
        do_ga=True,
        mu=0.04,
        u_max=2,
        metrics_trial_frequency=10)

    # Explore the environment
    agent1 = ACS2(cfg)
    population, explore_metrics = agent1.explore(grid, 1000, decay=False)

    for cl in sorted(population, key=lambda c: -c.fitness):
        if cl.does_anticipate_change():
            print_cl(cl)

    # Exploit
    agent2 = ACS2(cfg, population)
    pop_exploit, metric_exploit = agent2.exploit(grid, 100)

    # Print classifiers
    for cl in sorted(pop_exploit, key=lambda c: -c.fitness):
        if cl.does_anticipate_change():
            print_cl(cl)
