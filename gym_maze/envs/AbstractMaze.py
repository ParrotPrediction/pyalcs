import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_maze import Maze, WALL_MAPPING

import numpy as np
import logging
import random


logger = logging.getLogger(__name__)

ACTION_LOOKUP = {
    0: 'N',
    1: 'NE',
    2: 'E',
    3: 'SE',
    4: 'S',
    5: 'SW',
    6: 'W',
    7: 'NW'
}


class AbstractMaze(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, matrix):
        self.maze = Maze(matrix)

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Discrete(8)

    def _step(self, action):
        previous_observation = self._observe()
        logger.debug("Previous observation: [{}]".format(previous_observation))
        self._take_action(action, previous_observation)

        observation = self._observe()
        reward = self._get_reward()
        episode_over = self._is_over()

        return observation, reward, episode_over, {}

    def _reset(self):
        logger.debug("Resetting the environment")
        self._insert_animat()
        return self._observe()

    def _render(self, mode='human', close=False):
        logging.debug("Rendering the environment")
        ANIMAT_MARKER = '5'

        situation = np.copy(self.maze.matrix)
        situation[self.pos_y, self.pos_x] = ANIMAT_MARKER

        logger.info("\n{}".format(situation))

    def _observe(self):
        return self.maze.perception(self.pos_x, self.pos_y)

    def _get_reward(self):
        if self.maze.is_reward(self.pos_x, self.pos_y):
            return 1000

        return 0

    def _is_over(self):
        return self.maze.is_reward(self.pos_x, self.pos_y)

    def _take_action(self, action, observation):
        """Executes the action inside the maze"""
        animat_moved = False
        action_type = ACTION_LOOKUP[action]

        if action_type == "N" and not self.is_wall(observation[0]):
            self.pos_y -= 1
            animat_moved = True

        if action_type == 'NE' and not self.is_wall(observation[1]):
            self.pos_x += 1
            self.pos_y -= 1
            animat_moved = True

        if action_type == "E" and not self.is_wall(observation[2]):
            self.pos_x += 1
            animat_moved = True

        if action_type == 'SE' and not self.is_wall(observation[3]):
            self.pos_x += 1
            self.pos_y += 1
            animat_moved = True

        if action_type == "S" and not self.is_wall(observation[4]):
            self.pos_y += 1
            animat_moved = True

        if action_type == 'SW' and not self.is_wall(observation[5]):
            self.pos_x -= 1
            self.pos_y += 1
            animat_moved = True

        if action_type == "W" and not self.is_wall(observation[6]):
            self.pos_x -= 1
            animat_moved = True

        if action_type == 'NW' and not self.is_wall(observation[7]):
            self.pos_x -= 1
            self.pos_y -= 1
            animat_moved = True

        return animat_moved

    def _insert_animat(self):
        possible_coords = self.maze.get_possible_insertion_coordinates()

        starting_position = random.choice(possible_coords)
        self.pos_x = starting_position[0]
        self.pos_y = starting_position[1]

    @staticmethod
    def is_wall(perception):
        return perception == str(WALL_MAPPING)
