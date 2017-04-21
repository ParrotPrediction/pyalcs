import logging
import random
from collections import namedtuple
from os.path import dirname, abspath, join

from alcs.environment.Environment import Environment
from alcs.agent.Perception import Perception
from alcs.environment.maze import MazeMapping, MazeAction

logger = logging.getLogger(__name__)

# Default perception
MazePerception = namedtuple('MazePerception',
                            ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])


class Maze(Environment):
    def __init__(self, maze_file: str):
        """
        Initiate a new maze environment
        """
        self.mapping = MazeMapping()

        # Animat settings
        self.animat_pos_x = None
        self.animat_pos_y = None
        self.animat_moved = False

        self.max_x, self.max_y, self.matrix = None, None, None
        self._load_maze_from_file(maze_file)

    def move_was_successful(self) -> bool:
        """
        Checks if animat moved in this turn.

        :return: True is animat moved, false otherwise
        """
        return self.animat_moved

    def trial_finished(self) -> bool:
        """
        Gives information whether an animat is still searching the reward.

        :return: True if reward is found, false otherwise
        """
        return self.animat_found_reward

    def _load_maze_from_file(self, fname: str) -> None:
        """
        Reads maze definition from text file
        and creates required internal variables (maze dimensions,
        and it's matrix representation)

        :param fname: location of .maze file
        """
        basepath = dirname(__file__)
        filepath = abspath(join(basepath, '..', '..', '..', fname))

        with open(filepath) as file:
            max_x = int(file.readline())
            max_y = int(file.readline())
            matrix = [[None for x in range(max_x)] for y in range(max_y)]

            for y in range(0, max_y):
                row = file.readline()
                for x in range(0, max_x):
                    matrix[y][x] = self.mapping.find_value(row[x])

            logger.debug('Maze (%d x %d) imported successfully from '
                         'file [%s].', max_x, max_y, fname)

            self.max_x, self.max_y, self.matrix = max_x, max_y, matrix

    def insert_animat(self, pos_x: int = None, pos_y: int = None) -> None:
        self.animat_found_reward = False

        if pos_x is not None and pos_y is not None:
            if (not self._within_x_range(pos_x) or
                    not self._within_y_range(pos_y)):
                raise ValueError('Values outside allowed range')

            if not self.is_path(pos_x, pos_y):
                raise ValueError('Animat must be inserted into path')

            self.animat_pos_x = pos_x
            self.animat_pos_y = pos_y

            logger.info('Animat [(%d, %d)] placed into fixed initial cords',
                        pos_x, pos_y)
        else:
            possible_coords = self.get_possible_agent_insertion_coordinates()

            starting_position = random.choice(possible_coords)
            self.animat_pos_x = starting_position[0]
            self.animat_pos_y = starting_position[1]

            logger.info('Animat [(%d, %d)] placed into random initial cords',
                        self.animat_pos_x, self.animat_pos_y)

    def get_animat_perception(self, pos_x=None, pos_y=None) -> Perception:
        """
        :param pos_x:
        :param pos_y:
        :return: Animat Perception namedtuple object
        """

        if pos_x is None:
            pos_x = self.animat_pos_x

        if pos_y is None:
            pos_y = self.animat_pos_y

        if not self._within_x_range(pos_x):
            raise ValueError('X position not within allowed range')

        if not self._within_y_range(pos_y):
            raise ValueError('Y position not within allowed range')

        # Position N
        if pos_y == 0:
            n = None
        else:
            n = self.matrix[pos_y - 1][pos_x]

        # Position NE
        if pos_x == self.max_x - 1 or pos_y == 0:
            ne = None
        else:
            ne = self.matrix[pos_y - 1][pos_x + 1]

        # Position E
        if pos_x == self.max_x - 1:
            e = None
        else:
            e = self.matrix[pos_y][pos_x + 1]

        # Position SE
        if pos_x == self.max_x - 1 or pos_y == self.max_y - 1:
            se = None
        else:
            se = self.matrix[pos_y + 1][pos_x + 1]

        # Position S
        if pos_y == (self.max_y - 1):
            s = None
        else:
            s = self.matrix[pos_y + 1][pos_x]

        # Position SW
        if pos_x == 0 or pos_y == self.max_y - 1:
            sw = None
        else:
            sw = self.matrix[pos_y + 1][pos_x - 1]

        # Position W
        if pos_x == 0:
            w = None
        else:
            w = self.matrix[pos_y][pos_x - 1]

        # Position NW
        if pos_x == 0 or pos_y == 0:
            nw = None
        else:
            nw = self.matrix[pos_y - 1][pos_x - 1]

        maze_perception = MazePerception(
            N=n, NE=ne, E=e, SE=se, S=s, SW=sw, W=w, NW=nw)

        logger.debug('Animat [(%d, %d)] perception: [%s]',
                     pos_x, pos_y, maze_perception)

        return Perception(maze_perception)

    def execute_action(self,
                       action_value: int,
                       perception: Perception = None) -> int:
        """
        Orders animat to execute the action. The animat is not allowed
        to move into the wall. If animat didn't moved in this turn he receives
        no reward.

        :param action_value: value of action to execute
        :param perception: optional perception to test. If not specified
        current animat perception is used
        :return: reward for new position
        """
        if perception is None:
            perception = self.get_animat_perception()

        action = MazeAction().find_name(action_value)

        m_perception = MazePerception(perception)

        reward = 0
        self.animat_found_reward = False
        self.animat_moved = False
        logger.debug('Animat [(%d, %d)] ordered to execute action: [%s]',
                     self.animat_pos_x, self.animat_pos_y, action)

        if (action == "N" and
                self.not_wall(m_perception.N) and
                self._within_y_range()):

            self.animat_pos_y -= 1
            self.animat_moved = True

        if (action == 'NE' and
                self.not_wall(m_perception.N) and
                self.not_wall(m_perception.E) and
                self._within_x_range() and
                self._within_y_range()):

            self.animat_pos_x += 1
            self.animat_pos_y -= 1
            self.animat_moved = True

        if (action == "E" and
                self.not_wall(m_perception.E) and
                self._within_x_range()):

            self.animat_pos_x += 1
            self.animat_moved = True

        if (action == 'SE' and
                self.not_wall(m_perception.S) and
                self.not_wall(m_perception.E) and
                self._within_x_range() and
                self._within_y_range()):

            self.animat_pos_x += 1
            self.animat_pos_y += 1
            self.animat_moved = True

        if (action == "S" and
                self.not_wall(m_perception.S) and
                self._within_y_range()):

            self.animat_pos_y += 1
            self.animat_moved = True

        if (action == 'SW' and
                self.not_wall(m_perception.S) and
                self.not_wall(m_perception.W) and
                self._within_x_range() and
                self._within_y_range()):

            self.animat_pos_x -= 1
            self.animat_pos_y += 1
            self.animat_moved = True

        if (action == "W" and
                self.not_wall(m_perception.W) and
                self._within_x_range()):

            self.animat_pos_x -= 1
            self.animat_moved = True

        if (action == 'NW' and
                self.not_wall(m_perception.N) and
                self.not_wall(m_perception.W) and
                self._within_x_range() and
                self._within_y_range()):

            self.animat_pos_x -= 1
            self.animat_pos_y -= 1
            self.animat_moved = True

        if self.animat_moved:
            reward = self._calculate_reward()
        else:
            logger.debug('Animat [(%d, %d)] did not move.',
                         self.animat_pos_x, self.animat_pos_y)

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

    def _calculate_reward(self, pos_x: int = None, pos_y: int = None) -> int:
        """
        Calculates reward for given state

        :return: obtained reward
        """
        if pos_x is None and pos_y is None:
            pos_x = self.animat_pos_x
            pos_y = self.animat_pos_y

        reward = 0
        position_value = self._get_animat_position_value(pos_x, pos_y)

        if position_value == self.mapping['reward']['value']:
            reward = 1000
            self.animat_found_reward = True
            logger.info('*** ANIMAT FOUND REWARD! ***')

        if position_value == self.mapping['path']['value']:
            reward = 0

        logger.debug('Animat [(%d, %d)] received reward for position: [%d]',
                     pos_x, pos_y, reward)

        return reward

    def get_possible_agent_insertion_coordinates(self) -> list:
        """
        Returns a list with coordinates in the environment where
        an agent can be placed (only on the path).

        :return: list of tuples (X,Y) containing coordinates
        """
        possible_cords = []
        for x in range(0, self.max_x):
            for y in range(0, self.max_y):
                if self.is_path(x, y):
                    possible_cords.append((x, y))

        return possible_cords

    @staticmethod
    def get_possible_neighbour_cords(pos_x, pos_y) -> tuple:
        """
        Returns a tuple with coordinates for
        N, NE, E, SE, S, SW, W, NW neighbouring cells.
        """
        n = (pos_x, pos_y - 1)
        ne = (pos_x + 1, pos_y - 1)
        e = (pos_x + 1, pos_y)
        se = (pos_x + 1, pos_y + 1)
        s = (pos_x, pos_y + 1)
        sw = (pos_x - 1, pos_y + 1)
        w = (pos_x - 1, pos_y)
        nw = (pos_x - 1, pos_y - 1)

        return n, ne, e, se, s, sw, w, nw

    def is_wall(self, pos_x, pos_y):
        return (self.matrix[pos_y][pos_x] ==
                MazeMapping().mapping['wall']['value'])

    def is_path(self, pos_x, pos_y):
        return (self.matrix[pos_y][pos_x] ==
                MazeMapping().mapping['path']['value'])

    def is_reward(self, pos_x, pos_y):
        return (self.matrix[pos_y][pos_x] ==
                MazeMapping().mapping['reward']['value'])

    @staticmethod
    def not_wall(perceptron):
        return perceptron != MazeMapping().mapping['wall']['value']

    @staticmethod
    def moved_north(start, destination) -> bool:
        """
        :param start: start (X, Y) coordinates tuple
        :param destination: destination (X, Y) coordinates tuple
        :return: true if it was north move
        """
        return destination[1] + 1 == start[1]

    @staticmethod
    def moved_east(start, destination) -> bool:
        """
        :param start: start (X, Y) coordinates tuple
        :param destination: destination (X, Y) coordinates tuple
        :return: true if it was east move
        """
        return destination[0] - 1 == start[0]

    @staticmethod
    def moved_south(start, destination) -> bool:
        """
        :param start: start (X, Y) coordinates tuple
        :param destination: destination (X, Y) coordinates tuple
        :return: true if it was south move
        """
        return destination[1] - 1 == start[1]

    @staticmethod
    def moved_west(start, destination) -> bool:
        """
        :param start: start (X, Y) coordinates tuple
        :param destination: destination (X, Y) coordinates tuple
        :return: true if it was west move
        """
        return destination[0] + 1 == start[0]

    def _within_x_range(self, pos_x=None):
        if pos_x is None:
            pos_x = self.animat_pos_x

        return 0 <= pos_x < self.max_x

    def _within_y_range(self, pos_y=None):
        if pos_y is None:
            pos_y = self.animat_pos_y

        return 0 <= pos_y < self.max_y
