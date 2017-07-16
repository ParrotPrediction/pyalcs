import unittest

from alcs.environment.maze import Maze
from alcs.helpers.maze_utils import get_all_possible_transitions,\
    calculate_optimal_path_length


class MazeUtilsTest(unittest.TestCase):

    def setUp(self):
        self.env = Maze('tests/maze/test1.maze')

    def test_should_return_all_transitions(self):
        transitions = get_all_possible_transitions(self.env)

        for start, action, destination in transitions:
            print("{} => {} => {}".format(start, action, destination))

    def test_should_calculate_optimal_length(self):
        self.assertEqual(4.875, calculate_optimal_path_length(self.env))
