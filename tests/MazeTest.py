import unittest
from environment import Maze


class MazeTestCase(unittest.TestCase):

    def setUp(self):
        self.env = Maze('../mazes/m1.maze')

    def test_animat_perception(self):
        # Edge conditions
        self.assertListEqual([None, None, '#', '#'], self.env.get_animat_perception(0, 0))
        self.assertListEqual(['#', None, '#', '#'], self.env.get_animat_perception(0, 3))
        self.assertListEqual(['#', None, None, '#'], self.env.get_animat_perception(0, 7))
        self.assertListEqual([None, '#', '#', None], self.env.get_animat_perception(7, 0))
        self.assertListEqual(['.', '#', None, '#'], self.env.get_animat_perception(4, 7))

        # Good cases
        self.assertListEqual(['.', '#', '#', '#'], self.env.get_animat_perception(2, 5))
        self.assertListEqual(['.', '.', '#', '.'], self.env.get_animat_perception(5, 6))
        self.assertListEqual(['#', '.', '#', '.'], self.env.get_animat_perception(3, 4))

        # Wrong input values
        self.assertRaises(TypeError, self.env.get_animat_perception, -2, 4)
        self.assertRaises(TypeError, self.env.get_animat_perception, 2, 9)

    def test_get_animat_position_value(self):
        self.assertEqual('#', self.env.get_animat_position_value(0, 0))
        self.assertEqual('$', self.env.get_animat_position_value(1, 1))
        self.assertEqual('.', self.env.get_animat_position_value(4, 3))
        self.assertEqual('.', self.env.get_animat_position_value(3, 4))
        self.assertEqual('#', self.env.get_animat_position_value(5, 3))
        self.assertEqual('#', self.env.get_animat_position_value(3, 5))
        self.assertEqual('.', self.env.get_animat_position_value(5, 5))


if __name__ == '__main__':
    unittest.main()
