import logging
import pickle

import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer

from lcs.agents.acs2 import ACS2, Configuration
from lcs.agents import EnvironmentAdapter


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class MultiplexerAdapter(EnvironmentAdapter):
    @classmethod
    def to_genotype(cls, env_state):
        return [str(x) for x in env_state]


def get_actors():
    mp = gym.make('boolean-multiplexer-6bit-v0')
    cfg = Configuration(
        mp.env.observation_space.n, 2,
        environment_adapter=MultiplexerAdapter(),
        do_ga=True)

    return ACS2(cfg), mp


def dump_data(population, metrics, env, trials):
    logger.info("Dumping data to files")
    env_name = env.spec._env_name
    pickle.dump(population, open(f"population_{env_name}_{trials}.p", "wb"))
    pickle.dump(metrics, open(f"metrics_{env_name}_{trials}.p", "wb"))


if __name__ == '__main__':
    TRIALS = 5_000
    DUMP_DATA_TO_FILE = True

    agent, env = get_actors()

    population, metrics = agent.explore_exploit(env, TRIALS)
    logger.info("Experiment completed")

    reliable_classifiers = [c for c in population if c.is_reliable()]
    reliable_classifiers = sorted(reliable_classifiers, key=lambda cl: -cl.q)

    # Print top 20 reliable classifiers
    for cl in reliable_classifiers[:20]:
        print(f"{cl}, q: {cl.q:.2f}, exp: {cl.exp:.2f}")

    if DUMP_DATA_TO_FILE:
        dump_data(population, metrics, env, TRIALS)
