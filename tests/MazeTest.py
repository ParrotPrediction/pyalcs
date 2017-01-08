import unittest
from environment import Maze
from environment import MazeSymbols as MS


class MazeTestCase(unittest.TestCase):

    def setUp(self):
        self.env = Maze('../mazes/m1.maze')

    def test_should_load_maze_from_file(self):
        self.assertEqual(8, self.env.max_x)
        self.assertEqual(8, self.env.max_y)
        self.assertEqual(MS.REWARD.value, self.env.matrix[1][1])

    def test_animat_perception(self):
        # Edge conditions
        self.assertListEqual([None, None, MS.WALL.value, MS.WALL.value], self.env.get_animat_perception(0, 0))
        self.assertListEqual([MS.WALL.value, None, MS.WALL.value, MS.WALL.value], self.env.get_animat_perception(0, 3))
        self.assertListEqual([MS.WALL.value, None, None, MS.WALL.value], self.env.get_animat_perception(0, 7))
        self.assertListEqual([None, MS.WALL.value, MS.WALL.value, None], self.env.get_animat_perception(7, 0))
        self.assertListEqual([MS.PATH.value, MS.WALL.value, None, MS.WALL.value], self.env.get_animat_perception(4, 7))

        # Good cases
        self.assertListEqual([MS.PATH.value, MS.WALL.value, MS.WALL.value, MS.WALL.value], self.env.get_animat_perception(2, 5))
        self.assertListEqual([MS.PATH.value, MS.PATH.value, MS.WALL.value, MS.PATH.value], self.env.get_animat_perception(5, 6))
        self.assertListEqual([MS.WALL.value, MS.PATH.value, MS.WALL.value, MS.PATH.value], self.env.get_animat_perception(3, 4))

        # Wrong input values
        self.assertRaises(TypeError, self.env.get_animat_perception, -2, 4)
        self.assertRaises(TypeError, self.env.get_animat_perception, 2, 9)

    def test_get_animat_position_value(self):
        self.assertEqual(MS.WALL.value, self.env._get_animat_position_value(0, 0))
        self.assertEqual(MS.REWARD.value, self.env._get_animat_position_value(1, 1))
        self.assertEqual(MS.PATH.value, self.env._get_animat_position_value(4, 3))
        self.assertEqual(MS.PATH.value, self.env._get_animat_position_value(3, 4))
        self.assertEqual(MS.WALL.value, self.env._get_animat_position_value(5, 3))
        self.assertEqual(MS.WALL.value, self.env._get_animat_position_value(3, 5))
        self.assertEqual(MS.PATH.value, self.env._get_animat_position_value(5, 5))

    def test_should_insert_animat_randomly(self):
        for i in range(0, 100):
            self.env.insert_animat()
            self.assertTrue(Maze.not_wall(self.env._get_animat_position_value()))

    def test_should_insert_animat(self):
        # Check for wrong values, cannot insert into wall
        pass

    def test_should_execute_action(self):
        # check if coordinates changed
        # perception changed
        # reward is correct
        # illegal action protection
        pass


if __name__ == '__main__':
    unittest.main()
