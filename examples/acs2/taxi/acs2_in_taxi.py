import logging
from typing import Dict

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


def taxi_knowledge(population, environment) -> Dict:
    """
    Analyzes all possible transition in taxi environment and checks if there
    is a reliable classifier for it.

    Parameters
    ----------
    population
        list of classifiers
    environment
        taxi environment

    Returns
    -------
    Dict
        knowledge - percentage of transitions we are able to anticipate
            correctly (max 100)
    """
    transitions = environment.env.P

    # Take into consideration only reliable classifiers
    reliable_classifiers = [c for c in population if c.is_reliable()]

    # Count how many transitions are anticipated correctly
    nr_correct = 0

    # For all possible destinations from each path cell
    for start in range(500):
        for action in range(6):
            local_transitions = transitions[start][action]
            if len(local_transitions) > 1:
                pass

            prob, end, reward, done = local_transitions[0]

            p0 = (str(start), )
            p1 = (str(end), )

            if any([True for cl in reliable_classifiers
                    if cl.predicts_successfully(p0, action, p1)]):
                nr_correct += 1

    return {
            'knowledge': nr_correct / (500.0 * 6) * 100.0
    }


def taxi_metrics(population, environment):
    return {
        'agent': population_metrics(population, environment),
        'knowledge': taxi_knowledge(population, environment)
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
