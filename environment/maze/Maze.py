import logging
import random
from os.path import dirname, abspath, join
from enum import Enum, unique

from environment.Environment import Environment
from environment.maze.MazeMapping import MazeMapping

logger = logging.getLogger(__name__)


@unique
class MazeAction(Enum):
    """
    Represents possible animat action. Correlated with perception.
    """
    LEFT = 0
    TOP = 1
    RIGHT = 2
    DOWN = 3


class Maze(Environment):

    def __init__(self, maze_file):
        """
        Initiate a new maze environment
        """
        self.mapping = MazeMapping()

        # Animat settings
        self.animat_pos_x = None
        self.animat_pos_y = None
        self.animat_found_reward = False

        self.max_x, self.max_y, self.matrix = None, None, None
        self._load_maze_from_file(maze_file)

    def reset_animat_state(self):
        logger.debug('Resetting animat state')
        self.animat_pos_x = None
        self.animat_pos_y = None
        self.animat_found_reward = False

    def _load_maze_from_file(self, fname):
        """
        Reads maze definition from text file to create required internal variables

        :param fname: location of .maze file
        :return: dimension and maze matrix
        """
        basepath = dirname(__file__)
        filepath = abspath(join(basepath, '..', '..', fname))

        with open(filepath) as file:
            max_x = int(file.readline())
            max_y = int(file.readline())
            matrix = [[None for x in range(max_x)] for y in range(max_y)]

            for y in range(0, max_y):
                row = file.readline()
                for x in range(0, max_x):
                    matrix[y][x] = self.mapping.find_value(row[x])

            logger.debug('Maze (%d x %d) imported successfully from file [%s].', max_x, max_y, fname)
            self.max_x, self.max_y, self.matrix = max_x, max_y, matrix

    def insert_animat(self, pos_x=None, pos_y=None):
        if pos_x is not None and pos_y is not None:
            if not self._within_x_range(pos_x) or not self._within_y_range(pos_y):
                raise ValueError('Values outside allowed range')

            if self.matrix[pos_y][pos_x] != self.mapping.mapping['path']['value']:
                raise ValueError('Animat must be inserted into path')

            self.animat_pos_x = pos_x
            self.animat_pos_y = pos_y
            logger.info('Animat [(%d, %d)] placed into fixed initial cords', pos_x, pos_y)
        else:
            possible_coords = []
            for x in range(0, self.max_x):
                for y in range(0, self.max_y):
                    if self.matrix[x][y] == self.mapping.mapping['path']['value']:
                        possible_coords.append((x, y))

            starting_position = random.choice(possible_coords)
            self.animat_pos_x = starting_position[1]
            self.animat_pos_y = starting_position[0]

            logger.debug('Animat [(%d, %d)] placed into random initial cords', self.animat_pos_x, self.animat_pos_y)

    def get_animat_perception(self, pos_x=None, pos_y=None):
        """
        :param pos_x:
        :param pos_y:
        :return: [TOP, LEFT, BOTTOM, RIGHT]
        """

        if pos_x is None:
            pos_x = self.animat_pos_x

        if pos_y is None:
            pos_y = self.animat_pos_y

        if not self._within_x_range(pos_x):
            raise ValueError('X position not within allowed range')

        if not self._within_y_range(pos_y):
            raise ValueError('Y position not within allowed range')

        if pos_y == 0:
            top = None
        else:
            top = self.matrix[pos_y - 1][pos_x]

        if pos_x == 0:
            left = None
        else:
            left = self.matrix[pos_y][pos_x - 1]

        if pos_y == self.max_y - 1:
            bottom = None
        else:
            bottom = self.matrix[pos_y + 1][pos_x]

        if pos_x == self.max_x - 1:
            right = None
        else:
            right = self.matrix[pos_y][pos_x + 1]

        perception_symbols = [top, left, bottom, right]

        logger.debug('Animat [(%d, %d)] perception (TLBR): [%s]', self.animat_pos_x, self.animat_pos_y, perception_symbols)

        return perception_symbols

    def execute_action(self, perception: list, action: MazeAction):
        reward = 0
        animat_moved = False
        logger.debug('Animat [(%d, %d)] ordered to execute action: [%s]', self.animat_pos_x, self.animat_pos_y, action.name)

        if action == MazeAction.TOP and self.not_wall(perception[0]) and self._within_y_range():
            self.animat_pos_y += -1
            animat_moved = True

        if action == MazeAction.LEFT and self.not_wall(perception[1]) and self._within_x_range():
            self.animat_pos_x += -1
            animat_moved = True

        if action == MazeAction.DOWN and self.not_wall(perception[2]) and self._within_y_range():
            self.animat_pos_y += 1
            animat_moved = True

        if action == MazeAction.RIGHT and self.not_wall(perception[3]) and self._within_x_range():
            self.animat_pos_x += 1
            animat_moved = True

        if animat_moved:
            reward = self._calculate_reward()
        else:
            logger.debug('Animat [(%d, %d)] did not move.', self.animat_pos_x, self.animat_pos_y,)

        return reward

    def _get_animat_position_value(self, pos_x=None, pos_y=None):
        """
        Gets the value of the animat position. If no arguments
        are given returns the current position.

        :return: the value of the value of the position in the maze
        """
        if pos_x is None and pos_y is None:
            pos_x = self.animat_pos_x
            pos_y = self.animat_pos_y

        return self.matrix[pos_y][pos_x]

    def _calculate_reward(self, pos_x=None, pos_y=None):
        """
        Calculates reward for given state

        :return: obtained reward
        """
        if pos_x is None and pos_y is None:
            pos_x = self.animat_pos_x
            pos_y = self.animat_pos_y

        reward = None
        position_value = self._get_animat_position_value(pos_x, pos_y)

        if position_value == self.mapping.mapping['reward']['value']:
            reward = 2000
            self.animat_found_reward = True
            logger.info('Animat found reward!')

        if position_value == self.mapping.mapping['path']['value']:
            reward = 1

        logger.debug('Animat [(%d, %d)] received reward for position: [%d]', pos_x, pos_y, reward)
        return reward

    @staticmethod
    def not_wall(perceptron):
        return perceptron != MazeMapping().mapping['wall']['value']

    def _within_x_range(self, pos_x=None):
        if pos_x is None:
            pos_x = self.animat_pos_x

        return 0 <= pos_x < self.max_x

    def _within_y_range(self, pos_y=None):
        if pos_y is None:
            pos_y = self.animat_pos_y

        return 0 <= pos_y < self.max_y
