import unittest

from alcs.environment.maze import MazeActionC


class MazeActionTest(unittest.TestCase):

    def setUp(self):
        self.actions = MazeActionC()

    def test_get_all_actions(self):
        all_actions = self.actions.get_all_values()
        self.assertEquals(4, len(all_actions))

    def test_should_find_symbol(self):
        self.assertEquals('N', self.actions.find_symbol(1))
        self.assertRaises(ValueError, self.actions.find_symbol, 15)
