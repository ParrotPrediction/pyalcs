import os

import gym
# noinspection PyUnresolvedReferences
import gym_maze

from lcs import run_experiment
from lcs.agents.acs2 import ClassifiersList
from lcs.strategies.action_selection import choose_action


class Acs2Agent:

    def __init__(self, num_actions, epsilon):
        self.eval_mode = False
        self.population = ClassifiersList()
        self.match_set = None
        self._num_actions = num_actions
        self._epsilon = epsilon

    def begin_episode(self, unused_observation):
        self.match_set = ClassifiersList()
        return self._choose_action()

    def end_episode(self, unused_reward):
        pass

    def step(self, reward, observation):
        self.match_set = self.population.form_match_set(observation)
        return self._choose_action()

    def _choose_action(self):
        prob = 1.0 if self.eval_mode else self._epsilon

        return choose_action(
                self.match_set,
                self._num_actions,
                prob)


def create_agent(environment):
    return Acs2Agent(num_actions=environment.action_space.n)


if __name__ == '__main__':

    BASE_PATH = "/tmp"
    GAME = 'MazeF2-v0'
    LOG_PATH = os.path.join(BASE_PATH, 'acs2', GAME)

    env = gym.make(GAME)
    runner = run_experiment.Runner(LOG_PATH,
                                   create_agent,
                                   environment=env.env,
                                   num_iterations=200,
                                   max_steps_per_episode=50)

    runner.run_experiment()
