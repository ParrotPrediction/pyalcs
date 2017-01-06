from environment.Environment import Environment
from environment.maze.MazeAction import MazeAction

import random


class Maze(Environment):

    def __init__(self, maze_file):
        """
        Initiate a new maze environment
        """
        self.max_x, self.max_y, self.matrix = self.__load_maze_from_file(maze_file)
        self.animat_pos_x, self.animat_pos_y = self.insert_animat()
        self.animat_perception = self.get_animat_perception()

    @staticmethod
    def __load_maze_from_file(fname):
        """
        Reads maze definition from text file and creates internal variables

        :param fname: location of .maze file
        :return: dimension and maze matrix
        """
        with open(fname) as file:
            max_x = int(file.readline())
            max_y = int(file.readline())
            matrix = [[None for x in range(max_x)] for y in range(max_y)]

            for y in range(0, max_y):
                row = file.readline()
                for x in range(0, max_x):
                    matrix[y][x] = row[x]

            return max_x, max_y, matrix

    def insert_animat(self, pos_x=None, pos_y=None):
        if pos_x is not None and pos_y is not None and self.matrix[pos_x][pos_y] != '.':
            return pos_x, pos_y

        possible_coords = []
        for x in range(0, self.max_x):
            for y in range(0, self.max_y):
                if self.matrix[x][y] == '.':
                    possible_coords.append((x, y))

        starting_position = random.choice(possible_coords)

        return starting_position[0], starting_position[1]

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

        if not self.within_x_range(pos_x):
            raise TypeError('X position not within allowed range')

        if not self.within_y_range(pos_y):
            raise TypeError('Y position not within allowed range')

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

        return [top, left, bottom, right]

    def get_reward(self):
        raise NotImplementedError()

    def get_animat_position_value(self, pos_x=None, pos_y=None):
        if pos_x is None and pos_y is None:
            pos_x = self.animat_pos_x
            pos_y = self.animat_pos_y

        return self.matrix[pos_y][pos_x]

    def execute_action(self, action: MazeAction):
        animat_moved = False

        if action == MazeAction.TOP and self.not_wall(self.animat_perception[0]) and self.within_y_range(self.animat_pos_y):
            self.animat_pos_y += -1
            animat_moved = True

        if action == MazeAction.LEFT and self.not_wall(self.animat_perception[1]) and self.within_x_range(self.animat_pos_x):
            self.animat_pos_x += -1
            animat_moved = True

        if action == MazeAction.DOWN and self.not_wall(self.animat_perception[2]) and self.within_y_range(self.animat_pos_y):
            self.animat_pos_y += 1
            animat_moved = True

        if action == MazeAction.RIGHT and self.not_wall(self.animat_perception[3]) and self.within_x_range(self.animat_pos_x):
            self.animat_pos_x += 1
            animat_moved = True

        return 1

    def not_wall(self, perceptron):
        return int(perceptron) > -1

    def within_x_range(self, pos_x):
        return 0 <= pos_x < self.max_x

    def within_y_range(self, pos_y):
        return 0 <= pos_y < self.max_y
