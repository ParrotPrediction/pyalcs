# Logger
import logging

import gym
import numpy as np

from lcs.agents import EnvironmentAdapter
from lcs.agents.acs2 import ACS2, Configuration
from lcs.metrics import population_metrics

logging.basicConfig(level=logging.INFO)


trials = 1000
bins = 14


# Utils
def print_cl(cl):
    action = None
    marked = ''

    if cl.action == 0:
        action = 'L'
    if cl.action == 1:
        action = 'R'

    if cl.is_marked():
        marked = '(*)'

    return (
        f"{cl.condition} - {action} - {cl.effect}"
        f"[fit: {cl.fitness:.3f}, r: {cl.r:.2f}, ir: {cl.ir:.2f}] {marked}")

# Declare environment
env = gym.make('CartPole-v0')

# Determine min/max values
steps = 20000

obs_arr = np.zeros((steps, 4))

for i in range(steps):
    env.reset()
    done = False
    while not done:
        obs, reward, done, _ = env.step(env.action_space.sample())
        obs_arr[i, :] = obs

env.close()

_high = obs_arr.max(axis=0)
_low = obs_arr.min(axis=0)

_range = _high - _low


# Agent configuration
class CartPoleAdapter(EnvironmentAdapter):
    BINS = bins

    @classmethod
    def to_genotype(cls, obs):
        return np.round(((obs - _low) / _range) * cls.BINS)\
            .astype(int)\
            .astype(str)\
            .tolist()


def avg_fitness(pop):
    return np.mean([cl.fitness for cl in pop if cl.is_reliable()])


# collect more metrics
def cp_metrics(pop, env):
    metrics = {}
    metrics['avg_fitness'] = avg_fitness(pop)
    metrics.update(population_metrics(pop, env))

    return metrics


cfg = Configuration(
    classifier_length=4,
    number_of_possible_actions=2,
    epsilon=0.9,
    beta=0.05,
    gamma=0.95,
    theta_exp=50,
    theta_ga=50,
    do_ga=True,
    mu=0.03,
    u_max=4,
    metrics_trial_frequency=5,
    user_metrics_collector_fcn=cp_metrics,
    environment_adapter=CartPoleAdapter)

if __name__ == '__main__':
    agent = ACS2(cfg)
    population_explore, metrics_explore = agent.explore(
        env, trials, decay=True)

    print(len(population_explore))

    for cl in sorted(population_explore, key=lambda cl: -cl.fitness)[:30]:
        print(print_cl(cl))
