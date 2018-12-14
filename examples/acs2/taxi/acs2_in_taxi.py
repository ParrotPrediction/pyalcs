import logging

import gym
# noinspection PyUnresolvedReferences
import sys
sys.path.append("/home/e-dzia/openai-envs/")
import gym_taxi_goal

from examples.acs2.taxi.TaxiAdapter import TaxiAdapter
from lcs.agents.acs2 import ACS2, Configuration

# Configure logger
from lcs.metrics import population_metrics

logging.basicConfig(level=logging.INFO)


def taxi_metrics(population, environment):
    return {
        'population': population_metrics(population, environment)
    }


if __name__ == '__main__':
    # Load desired environment
    environment = gym.make('TaxiGoal-v0')

    environment.reset()

    environment.render()

    # Configure and create the agent
    cfg = Configuration(1, 6,
                        epsilon=1.0,
                        do_ga=False,
                        environment_adapter=TaxiAdapter,
                        metrics_trial_frequency=1,
                        user_metrics_collector_fcn=taxi_metrics,
                        do_action_planning=True,
                        action_planning_frequency=50
                        )
    logging.info(cfg)

    # Explore the environment
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(environment, 1000)

    # Exploit the environment
    agent = ACS2(cfg, population)
    population, exploit_metric = agent.exploit(environment, 10)

    for metric in exploit_metric:
        logging.info(metric)
