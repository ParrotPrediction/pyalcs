import unittest

from alcs.agent import Perception
from alcs.agent.acs3 import Condition, PMark


class PMarkTest(unittest.TestCase):

    def setUp(self):
        self.mark = PMark()

    def test_should_initialize_mark(self):
        self.assertEqual(8, len(self.mark))

        for mark in self.mark:
            self.assertEqual(0, len(mark))

    def test_should_set_single_mark(self):
        self.mark[1] = '0'

        self.assertEqual(8, len(self.mark))
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
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        self.mark.set_mark(p0)

        for mark in self.mark:
            self.assertEqual(1, len(mark))

    def test_should_get_differences(self):
        self.mark[0] = '1'
        self.mark[1] = '0'
        self.mark[2] = '0'
        self.mark[3] = '0'
        self.mark[4] = '1'
        self.mark[5] = '0'
        self.mark[6] = '1'
        self.mark[7] = '1'

        p0 = Perception(['0', '0', '0', '0', '0', '0', '0', '1'])

        self.assertEqual(
            Condition(['#', '#', '#', '#', '0', '#', '#', '#']),
            self.mark.get_differences(p0))
