import unittest

from alcs.environment.maze import MazeAction


class MazeActionTest(unittest.TestCase):

    def setUp(self):
        self.actions = MazeAction()

    def test_get_all_actions(self):
        all_actions = self.actions.get_all_values()
        self.assertEquals(4, len(all_actions))

    def test_should_find_symbol(self):
        self.assertEquals('↑', self.actions.find_symbol(1))
        self.assertRaises(ValueError, self.actions.find_symbol, 15)

    def test_should_find_name(self):
        self.assertEquals("N", self.actions.find_name(1))
        self.assertRaises(ValueError, self.actions.find_name, 15)
