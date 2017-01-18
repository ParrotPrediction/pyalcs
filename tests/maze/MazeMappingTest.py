import unittest

from acs.environment.maze import MazeMapping


class MazeMappingTest(unittest.TestCase):

    def setUp(self):
        self.mapping = MazeMapping()

    def test_should_find_mapping_values(self):
        self.assertEqual(0, self.mapping.find_value('#'))
        self.assertEqual(1, self.mapping.find_value('.'))
        self.assertEqual(9, self.mapping.find_value('$'))
        self.assertRaises(ValueError, self.mapping.find_value, '*')
