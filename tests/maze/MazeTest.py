import unittest

from alcs.environment.maze import MazeAction
from alcs.environment.maze import Maze


class MazeTest(unittest.TestCase):

    def setUp(self):
        self.env = Maze('tests/maze/test1.maze')
        self.MA = MazeAction()

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
            [None,  # N
             None,  # NE
             self.env.mapping['wall']['value'],  # E
             self.env.mapping['reward']['value'],  # SE
             self.env.mapping['wall']['value'],  # S
             None,  # SW
             None,  # W
             None],  # NW
            list(self.env.get_animat_perception(0, 0))
        )

        self.assertListEqual(
            [self.env.mapping['wall']['value'],  # N
             self.env.mapping['wall']['value'],  # NE
             self.env.mapping['wall']['value'],  # E
             self.env.mapping['wall']['value'],  # SE
             self.env.mapping['wall']['value'],  # S
             None,  # SW
             None,  # W
             None],  # NW
            list(self.env.get_animat_perception(0, 3))
        )

        self.assertListEqual(
            [self.env.mapping['wall']['value'],  # N
             self.env.mapping['wall']['value'],  # NE
             self.env.mapping['wall']['value'],  # E
             None,  # SE
             None,  # S
             None,  # SW
             None,  # W
             None],  # NW
            list(self.env.get_animat_perception(0, 7))
        )

        self.assertListEqual(
            [None,  # N
             None,  # NE
             None,  # E
             None,  # SE
             self.env.mapping['wall']['value'],  # S
             self.env.mapping['wall']['value'],  # SW
             self.env.mapping['wall']['value'],  # W
             None],  # NW
            list(self.env.get_animat_perception(7, 0))
        )

        self.assertListEqual(
            [self.env.mapping['path']['value'],  # N
             self.env.mapping['path']['value'],  # NE
             self.env.mapping['wall']['value'],  # E
             None,  # SE
             None,  # S
             None,  # SW
             self.env.mapping['wall']['value'],  # W
             self.env.mapping['path']['value']],  # NW
            list(self.env.get_animat_perception(4, 7))
        )

        # Good cases
        self.assertListEqual(
            [self.env.mapping['path']['value'],  # N
             self.env.mapping['path']['value'],  # NE
             self.env.mapping['wall']['value'],  # E
             self.env.mapping['path']['value'],  # SE
             self.env.mapping['wall']['value'],  # S
             self.env.mapping['wall']['value'],  # SW
             self.env.mapping['wall']['value'],  # W
             self.env.mapping['wall']['value']],  # NW
            list(self.env.get_animat_perception(2, 5)))

        self.assertListEqual(
            [self.env.mapping['path']['value'],  # N
             self.env.mapping['wall']['value'],  # NE
             self.env.mapping['path']['value'],  # E
             self.env.mapping['wall']['value'],  # SE
             self.env.mapping['wall']['value'],  # S
             self.env.mapping['wall']['value'],  # SW
             self.env.mapping['path']['value'],  # W
             self.env.mapping['wall']['value']],  # NW
            list(self.env.get_animat_perception(5, 6)))

        self.assertListEqual(
            [self.env.mapping['wall']['value'],  # N
             self.env.mapping['path']['value'],  # NE
             self.env.mapping['path']['value'],  # E
             self.env.mapping['wall']['value'],  # SE
             self.env.mapping['wall']['value'],  # S
             self.env.mapping['path']['value'],  # SW
             self.env.mapping['path']['value'],  # W
             self.env.mapping['wall']['value']],  # NW
            list(self.env.get_animat_perception(3, 4)))

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
        self.assertListEqual(
            [self.env.mapping['wall']['value'],  # N
             self.env.mapping['wall']['value'],  # NE
             self.env.mapping['path']['value'],  # E
             self.env.mapping['wall']['value'],  # SE
             self.env.mapping['path']['value'],  # S
             self.env.mapping['wall']['value'],  # SW
             self.env.mapping['path']['value'],  # W
             self.env.mapping['wall']['value']],  # NW
            list(self.env.get_animat_perception())
        )

        # Wrong input values
        # Try to insert into wall
        self.assertRaises(ValueError, self.env.insert_animat, 4, 0)

        # Try to insert out of range
        self.assertRaises(ValueError, self.env.insert_animat, 9, 2)

    def test_should_execute_chain_of_actions(self):
        # Put animat in maze
        self.env.insert_animat(4, 2)

        # Make sure that the searching state is correct
        self.assertFalse(self.env.animat_found_reward)

        # Check if animat landed in desired position
        self.assertEqual(
            self.env.mapping['path']['value'],
            self.env._get_animat_position_value())

        # Check if perception is ok
        self.assertListEqual(
            [self.env.mapping['path']['value'],  # N
             self.env.mapping['path']['value'],  # NE
             self.env.mapping['wall']['value'],  # E
             self.env.mapping['wall']['value'],  # SE
             self.env.mapping['path']['value'],  # S
             self.env.mapping['wall']['value'],  # SW
             self.env.mapping['wall']['value'],  # W
             self.env.mapping['path']['value']],  # NW
            list(self.env.get_animat_perception()))

        # Tell animat to go up (should be ok)
        reward = self.env.execute_action(self.MA['N']['value'])

        # Validate if coordinates changed
        self.assertEqual(4, self.env.animat_pos_x)
        self.assertEqual(1, self.env.animat_pos_y)

        # Check if perception also changed
        self.assertListEqual(
            [self.env.mapping['wall']['value'],  # N
             self.env.mapping['wall']['value'],  # NE
             self.env.mapping['path']['value'],  # E
             self.env.mapping['wall']['value'],  # SE
             self.env.mapping['path']['value'],  # S
             self.env.mapping['wall']['value'],  # SW
             self.env.mapping['path']['value'],  # W
             self.env.mapping['wall']['value']],  # NW
            list(self.env.get_animat_perception()))

        # Make sure that the searching state is correct
        self.assertFalse(self.env.animat_found_reward)

        # Check if proper reward for visiting path was collected
        self.assertEqual(0, reward)

        # Now try to enter the wall (action - north)
        reward = self.env.execute_action(self.MA['N']['value'])

        # Validate if the coordinates did not changed
        self.assertEqual(4, self.env.animat_pos_x)
        self.assertEqual(1, self.env.animat_pos_y)

        # Check if no reward was collected
        self.assertEqual(0, reward)

        # Now let's go left / west (should be ok)
        reward = self.env.execute_action(self.MA['W']['value'])

        # Validate if the coordinates changed
        self.assertEqual(3, self.env.animat_pos_x)
        self.assertEqual(1, self.env.animat_pos_y)

        # Make sure that the searching state is correct
        self.assertFalse(self.env.animat_found_reward)

        # Check if proper reward for visiting path was collected
        self.assertEqual(0, reward)

        # Go left for the second time (should be ok)
        reward = self.env.execute_action(self.MA['W']['value'])

        # Validate if the coordinates changed
        self.assertEqual(2, self.env.animat_pos_x)
        self.assertEqual(1, self.env.animat_pos_y)

        # Check if proper reward for visiting path was collected
        self.assertEqual(0, reward)

        # Now the animat should see the final reward
        self.assertListEqual(
            [self.env.mapping['wall']['value'],  # N
             self.env.mapping['wall']['value'],  # NE
             self.env.mapping['path']['value'],  # E
             self.env.mapping['wall']['value'],  # SE
             self.env.mapping['wall']['value'],  # S
             self.env.mapping['wall']['value'],  # SW
             self.env.mapping['reward']['value'],  # W
             self.env.mapping['wall']['value']],  # NW
            list(self.env.get_animat_perception()))

        # Lets collect it by moving left for the third time (should be ok)
        reward = self.env.execute_action(self.MA['W']['value'])

        # Validate if the coordinates changed
        self.assertEqual(1, self.env.animat_pos_x)
        self.assertEqual(1, self.env.animat_pos_y)

        # Make sure that the searching state is correct
        self.assertTrue(self.env.animat_found_reward)

        # Check if proper reward for finding reward was collected
        self.assertEqual(1000, reward)

    def test_should_get_all_possible_insertion_coordinates(self):
        insertion_points = self.env.get_possible_agent_insertion_coordinates()
        self.assertEqual(16, len(insertion_points))

        # Check some paths randomly
        self.assertIn((2, 1), insertion_points)

        # These points are not "insertable"
        self.assertNotIn((1, 1), insertion_points)  # reward
        self.assertNotIn((2, 3), insertion_points)  # wall

    def test_should_return_neighbours_coordinates(self):
        neighbours = Maze.get_possible_neighbour_cords(5, 5)

        self.assertEqual(8, len(neighbours))
        self.assertEqual((5, 4), neighbours[0])  # N
        self.assertEqual((6, 4), neighbours[1])  # NE
        self.assertEqual((6, 5), neighbours[2])  # E
        self.assertEqual((6, 6), neighbours[3])  # SE
        self.assertEqual((5, 6), neighbours[4])  # S
        self.assertEqual((4, 6), neighbours[5])  # SW
        self.assertEqual((4, 5), neighbours[6])  # W
        self.assertEqual((4, 4), neighbours[7])  # NW

    def test_should_detect_movement_directions(self):
        self.assertTrue(Maze.moved_north((5, 5), (5, 4)))
        self.assertTrue(Maze.moved_east((5, 5), (6, 5)))
        self.assertTrue(Maze.moved_south((5, 5), (5, 6)))
        self.assertTrue(Maze.moved_west((5, 5), (4, 5)))

    def test_should_move_n(self):
        # Given
        maze = Maze('tests/maze/test2.maze')
        maze.insert_animat(2, 2)

        # When
        maze.execute_action(self.MA['N']['value'])

        # Then
        self.assertEqual(2, maze.animat_pos_x)
        self.assertEqual(1, maze.animat_pos_y)
        self.assertListEqual(
            [maze.mapping['wall']['value'],  # N
             maze.mapping['wall']['value'],  # NE
             maze.mapping['path']['value'],  # E
             maze.mapping['path']['value'],  # SE
             maze.mapping['path']['value'],  # S
             maze.mapping['path']['value'],  # SW
             maze.mapping['path']['value'],  # W
             maze.mapping['wall']['value']],  # NW
            list(maze.get_animat_perception()))

    def test_should_move_ne(self):
        # Given
        maze = Maze('tests/maze/test2.maze')
        maze.insert_animat(2, 2)

        # When
        maze.execute_action(self.MA['NE']['value'])

        # Then
        self.assertEqual(3, maze.animat_pos_x)
        self.assertEqual(1, maze.animat_pos_y)
        self.assertListEqual(
            [maze.mapping['wall']['value'],  # N
             maze.mapping['wall']['value'],  # NE
             maze.mapping['wall']['value'],  # E
             maze.mapping['wall']['value'],  # SE
             maze.mapping['path']['value'],  # S
             maze.mapping['path']['value'],  # SW
             maze.mapping['path']['value'],  # W
             maze.mapping['wall']['value']],  # NW
            list(maze.get_animat_perception()))

    def test_should_move_e(self):
        # Given
        maze = Maze('tests/maze/test2.maze')
        maze.insert_animat(2, 2)

        # When
        maze.execute_action(self.MA['E']['value'])

        # Then
        self.assertEqual(3, maze.animat_pos_x)
        self.assertEqual(2, maze.animat_pos_y)
        self.assertListEqual(
            [maze.mapping['path']['value'],  # N
             maze.mapping['wall']['value'],  # NE
             maze.mapping['wall']['value'],  # E
             maze.mapping['wall']['value'],  # SE
             maze.mapping['path']['value'],  # S
             maze.mapping['path']['value'],  # SW
             maze.mapping['path']['value'],  # W
             maze.mapping['path']['value']],  # NW
            list(maze.get_animat_perception()))

    def test_should_move_se(self):
        # Given
        maze = Maze('tests/maze/test2.maze')
        maze.insert_animat(2, 2)

        # When
        maze.execute_action(self.MA['SE']['value'])

        # Then
        self.assertEqual(3, maze.animat_pos_x)
        self.assertEqual(3, maze.animat_pos_y)
        self.assertListEqual(
            [maze.mapping['path']['value'],  # N
             maze.mapping['wall']['value'],  # NE
             maze.mapping['wall']['value'],  # E
             maze.mapping['wall']['value'],  # SE
             maze.mapping['wall']['value'],  # S
             maze.mapping['wall']['value'],  # SW
             maze.mapping['path']['value'],  # W
             maze.mapping['path']['value']],  # NW
            list(maze.get_animat_perception()))

    def test_should_move_s(self):
        # Given
        maze = Maze('tests/maze/test2.maze')
        maze.insert_animat(2, 2)

        # When
        maze.execute_action(self.MA['S']['value'])

        # Then
        self.assertEqual(2, maze.animat_pos_x)
        self.assertEqual(3, maze.animat_pos_y)
        self.assertListEqual(
            [maze.mapping['path']['value'],  # N
             maze.mapping['path']['value'],  # NE
             maze.mapping['path']['value'],  # E
             maze.mapping['wall']['value'],  # SE
             maze.mapping['wall']['value'],  # S
             maze.mapping['wall']['value'],  # SW
             maze.mapping['path']['value'],  # W
             maze.mapping['path']['value']],  # NW
            list(maze.get_animat_perception()))

    def test_should_move_sw(self):
        # Given
        maze = Maze('tests/maze/test2.maze')
        maze.insert_animat(2, 2)

        # When
        maze.execute_action(self.MA['SW']['value'])

        # Then
        self.assertEqual(1, maze.animat_pos_x)
        self.assertEqual(3, maze.animat_pos_y)
        self.assertListEqual(
            [maze.mapping['path']['value'],  # N
             maze.mapping['path']['value'],  # NE
             maze.mapping['path']['value'],  # E
             maze.mapping['wall']['value'],  # SE
             maze.mapping['wall']['value'],  # S
             maze.mapping['wall']['value'],  # SW
             maze.mapping['wall']['value'],  # W
             maze.mapping['wall']['value']],  # NW
            list(maze.get_animat_perception()))

    def test_should_move_w(self):
        # Given
        maze = Maze('tests/maze/test2.maze')
        maze.insert_animat(2, 2)

        # When
        maze.execute_action(self.MA['W']['value'])

        # Then
        self.assertEqual(1, maze.animat_pos_x)
        self.assertEqual(2, maze.animat_pos_y)
        self.assertListEqual(
            [maze.mapping['path']['value'],  # N
             maze.mapping['path']['value'],  # NE
             maze.mapping['path']['value'],  # E
             maze.mapping['path']['value'],  # SE
             maze.mapping['path']['value'],  # S
             maze.mapping['wall']['value'],  # SW
             maze.mapping['wall']['value'],  # W
             maze.mapping['wall']['value']],  # NW
            list(maze.get_animat_perception()))

    def test_should_move_nw(self):
        # Given
        maze = Maze('tests/maze/test2.maze')
        maze.insert_animat(2, 2)

        # When
        maze.execute_action(self.MA['NW']['value'])

        # Then
        self.assertEqual(1, maze.animat_pos_x)
        self.assertEqual(1, maze.animat_pos_y)
        self.assertListEqual(
            [maze.mapping['wall']['value'],  # N
             maze.mapping['wall']['value'],  # NE
             maze.mapping['path']['value'],  # E
             maze.mapping['path']['value'],  # SE
             maze.mapping['path']['value'],  # S
             maze.mapping['wall']['value'],  # SW
             maze.mapping['wall']['value'],  # W
             maze.mapping['wall']['value']],  # NW
            list(maze.get_animat_perception()))
