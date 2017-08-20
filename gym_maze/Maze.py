import logging

logger = logging.getLogger(__name__)

PATH_MAPPING = 0
WALL_MAPPING = 1
REWARD_MAPPING = 9


class Maze:
    """
    Creates new maze.
    
    Mapping:
    0 - path
    1 - wall
    9 - reward
    """

    def __init__(self, matrix):
        self.matrix = matrix
        self.max_x = self.matrix.shape[1]
        self.max_y = self.matrix.shape[0]

    def get_possible_insertion_coordinates(self):
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

    def perception(self, pos_x, pos_y):
        if not self._within_x_range(pos_x):
            raise ValueError('X position not within allowed range')

        if not self._within_y_range(pos_y):
            raise ValueError('Y position not within allowed range')

            # Position N
        if pos_y == 0:
            n = None
        else:
            n = str(self.matrix[pos_y - 1, pos_x])

            # Position NE
        if pos_x == self.max_x - 1 or pos_y == 0:
            ne = None
        else:
            ne = str(self.matrix[pos_y - 1, pos_x + 1])

            # Position E
        if pos_x == self.max_x - 1:
            e = None
        else:
            e = str(self.matrix[pos_y, pos_x + 1])

            # Position SE
        if pos_x == self.max_x - 1 or pos_y == self.max_y - 1:
            se = None
        else:
            se = str(self.matrix[pos_y + 1, pos_x + 1])

            # Position S
        if pos_y == (self.max_y - 1):
            s = None
        else:
            s = str(self.matrix[pos_y + 1, pos_x])

            # Position SW
        if pos_x == 0 or pos_y == self.max_y - 1:
            sw = None
        else:
            sw = str(self.matrix[pos_y + 1, pos_x - 1])

            # Position W
        if pos_x == 0:
            w = None
        else:
            w = str(self.matrix[pos_y, pos_x - 1])

            # Position NW
        if pos_x == 0 or pos_y == 0:
            nw = None
        else:
            nw = str(self.matrix[pos_y - 1, pos_x - 1])

        return n, ne, e, se, s, sw, w, nw

    def is_wall(self, pos_x, pos_y):
        return self.matrix[pos_y, pos_x] == WALL_MAPPING

    def is_path(self, pos_x, pos_y):
        return self.matrix[pos_y, pos_x] == PATH_MAPPING

    def is_reward(self, pos_x, pos_y):
        return self.matrix[pos_y, pos_x] == REWARD_MAPPING

    def _within_x_range(self, x):
        return 0 <= x < self.max_x

    def _within_y_range(self, y):
        return 0 <= y < self.max_y
