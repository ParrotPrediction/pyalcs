import unittest
from environment import Maze


class MazeTest(unittest.TestCase):

    def setUp(self):
        self.env = Maze('tests/maze/test1.maze')

    def test_should_load_maze_from_file(self):
        self.assertEqual(8, self.env.max_x)
        self.assertEqual(8, self.env.max_y)
        self.assertEqual(
            self.env.mapping['reward']['value'],
            self.env.matrix[1][1]
        )

    def test_animat_perception(self):
        # Edge conditions
        self.assertListEqual(
            [None,
             None,
             self.env.mapping['wall']['value'],
             self.env.mapping['wall']['value']],
            self.env.get_animat_perception(0, 0)
        )

        self.assertListEqual(
            [self.env.mapping['wall']['value'],
             None,
             self.env.mapping['wall']['value'],
             self.env.mapping['wall']['value']],
            self.env.get_animat_perception(0, 3)
        )

        self.assertListEqual(
            [self.env.mapping['wall']['value'],
             None,
             None,
             self.env.mapping['wall']['value']],
            self.env.get_animat_perception(0, 7)
        )

        self.assertListEqual(
            [None,
             self.env.mapping['wall']['value'],
             self.env.mapping['wall']['value'],
             None],
            self.env.get_animat_perception(7, 0)
        )

        self.assertListEqual(
            [self.env.mapping['path']['value'],
             self.env.mapping['wall']['value'],
             None,
             self.env.mapping['wall']['value']],
            self.env.get_animat_perception(4, 7)
        )

        # Good cases
        self.assertListEqual(
            [self.env.mapping['path']['value'],
             self.env.mapping['wall']['value'],
             self.env.mapping['wall']['value'],
             self.env.mapping['wall']['value']],
            self.env.get_animat_perception(2, 5))

        self.assertListEqual(
            [self.env.mapping['path']['value'],
             self.env.mapping['path']['value'],
             self.env.mapping['wall']['value'],
             self.env.mapping['path']['value']],
            self.env.get_animat_perception(5, 6))

        self.assertListEqual(
            [self.env.mapping['wall']['value'],
             self.env.mapping['path']['value'],
             self.env.mapping['wall']['value'],
             self.env.mapping['path']['value']],
            self.env.get_animat_perception(3, 4))

        # Wrong input values
        self.assertRaises(ValueError, self.env.get_animat_perception, -2, 4)
        self.assertRaises(ValueError, self.env.get_animat_perception, 2, 9)

    def test_get_animat_position_value(self):
        self.assertEqual(
            self.env.mapping['wall']['value'],
            self.env._get_animat_position_value(0, 0)
        )

        self.assertEqual(
            self.env.mapping['reward']['value'],
            self.env._get_animat_position_value(1, 1)
        )

        self.assertEqual(
            self.env.mapping['path']['value'],
            self.env._get_animat_position_value(4, 3)
        )

        self.assertEqual(
            self.env.mapping['path']['value'],
            self.env._get_animat_position_value(3, 4)
        )

        self.assertEqual(
            self.env.mapping['wall']['value'],
            self.env._get_animat_position_value(5, 3)
        )

        self.assertEqual(
            self.env.mapping['wall']['value'],
            self.env._get_animat_position_value(3, 5)
        )

        self.assertEqual(
            self.env.mapping['path']['value'],
            self.env._get_animat_position_value(5, 5)
        )

    def test_should_insert_animat_randomly(self):
        for i in range(0, 100):
            self.env.insert_animat()
            position_value = self.env._get_animat_position_value()
            self.assertTrue(Maze.not_wall(position_value))

    def test_should_insert_animat(self):
        self.env.insert_animat(4, 1)
        self.assertEqual(
            [self.env.mapping['wall']['value'],
             self.env.mapping['path']['value'],
             self.env.mapping['path']['value'],
             self.env.mapping['path']['value']],
            self.env.get_animat_perception()
        )

        # Wrong input values
        # Try to insert into wall
        self.assertRaises(ValueError, self.env.insert_animat, 4, 0)

        # Try to insert out of range
        self.assertRaises(ValueError, self.env.insert_animat, 9, 2)

    @unittest.skip("TODO")
    def test_should_execute_action(self):
        # check if coordinates changed
        # perception changed
        # reward is correct
        # illegal action protection
        pass


if __name__ == '__main__':
    unittest.main()
