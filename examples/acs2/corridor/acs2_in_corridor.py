import logging

import gym
# noinspection PyUnresolvedReferences
import gym_corridor

from lcs.agents import EnvironmentAdapter
from lcs.agents.acs2 import ACS2, Configuration

# Configure logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class CorridorAdapter(EnvironmentAdapter):
    @classmethod
    def to_genotype(cls, phenotype):
        return phenotype,


if __name__ == '__main__':
    # Load desired environment
    corridor = gym.make('corridor-20-v0')

    # Configure and create the agent
    cfg = Configuration(
        classifier_length=1,
        number_of_possible_actions=2,
        beta=0.03,
        gamma=0.97,
        theta_exp=50,
        theta_ga=50,
        do_ga=True,
        mu=0.02,
        u_max=1,
        metrics_trial_frequency=20,
        environment_adapter=CorridorAdapter
    )

    # Explore the environment
    logging.info("Exploring environment")
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(corridor, 1000)

    population = sorted(population, key=lambda cl: -cl.fitness)

    print("ok")

    for cl in population:
        print(cl)
