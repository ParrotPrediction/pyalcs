import unittest

from alcs.agent import Perception
from alcs.agent.acs2 import Condition, PMark


class PMarkTest(unittest.TestCase):

    def setUp(self):
        self.mark = PMark()

    def test_should_initialize_mark(self):
        self.assertEqual(0, len(self.mark))

        for mark in self.mark:
            self.assertEqual(0, len(mark))

    def test_should_detect_if_marked(self):
        self.assertTrue(self.mark.is_empty())

        # Add some mark
        self.mark[1] = '0'
        self.assertFalse(self.mark.is_empty())

    def test_should_set_single_mark(self):
        self.mark[1] = '0'

        self.assertEqual(1, len(self.mark))
        self.assertEqual(1, len(self.mark[1]))
        self.assertIn('0', self.mark[1])

        # Try to add the mark one more time into the same position
        self.mark[1] = '1'
        self.assertEqual(2, len(self.mark[1]))
        self.assertIn('0', self.mark[1])
        self.assertIn('1', self.mark[1])

        # Check if duplicates are avoided
        self.mark[1] = '1'
        self.assertEqual(2, len(self.mark[1]))
        self.assertIn('0', self.mark[1])
        self.assertIn('1', self.mark[1])

    def test_should_set_mark_from_perception(self):
        # Given
        p0 = Perception(['0', '1', '1', '1', '0', '1', '1', '1'])
        self.mark[0] = '1'
        self.mark[2] = '1'
        self.mark[3] = '1'
        self.mark[6] = '1'

        # When
        self.mark.set_mark(p0)

        # Then
        self.assertEqual(4, len(self.mark))

        self.assertEqual(2, len(self.mark[0]))
        self.assertIn('0', self.mark[0])
        self.assertIn('1', self.mark[0])

        self.assertEqual(1, len(self.mark[2]))
        self.assertIn('1', self.mark[2])

        self.assertEqual(1, len(self.mark[3]))
        self.assertIn('1', self.mark[3])

        self.assertEqual(1, len(self.mark[6]))
        self.assertIn('1', self.mark[6])
