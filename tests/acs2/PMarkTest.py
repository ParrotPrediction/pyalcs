import unittest

from alcs.acs2 import PMark

from alcs import Perception


class PMarkTest(unittest.TestCase):

    def setUp(self):
        self.mark = PMark()

    def test_should_initialize_mark(self):
        self.assertEqual(0, len(self.mark))

        for mark in self.mark:
            self.assertEqual(0, len(mark))

    def test_should_mark_with_non_string_char(self):
        self.assertRaises(TypeError, self.mark.__setitem__, 0, 1)

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

    def test_should_get_differences_1(self):
        # Given
        p0 = Perception(['0', '1', '1', '0', '0', '0', '1', '1'])

        # When
        diff = self.mark.get_differences(p0)

        # Then
        self.assertIsNone(diff)

    def test_should_get_differences_2(self):
        # Given
        p0 = Perception(['1', '1', '0', '1', '1', '1', '0', '1'])
        self.mark[0] = '1'
        self.mark[1] = '1'
        self.mark[2] = '0'
        self.mark[3] = '0'
        self.mark[4] = '0'
        self.mark[5] = '0'
        self.mark[6] = '1'
        self.mark[7] = '0'

        for _ in range(100):
            # When
            diff = self.mark.get_differences(p0)

            # Then
            self.assertIsNotNone(diff)
            self.assertEqual('#', diff[0])
            self.assertEqual('#', diff[1])
            self.assertEqual('#', diff[2])
            self.assertEqual(1, diff.specificity)

    def test_should_get_differences_3(self):
        # Given
        p0 = Perception(['0', '1', '1', '0', '0', '0', '0', '0'])
        self.mark[0].update(['0', '1'])
        self.mark[1].update(['1'])
        self.mark[2].update(['0', '1'])
        self.mark[3].update(['1'])
        self.mark[4].update(['0', '1'])
        self.mark[5].update(['1'])
        self.mark[6].update(['0', '1'])
        self.mark[7].update(['1'])

        for _ in range(100):
            # When
            diff = self.mark.get_differences(p0)

            # Then
            self.assertIsNotNone(diff)
            self.assertEqual('#', diff[0])
            self.assertEqual('#', diff[1])
            self.assertEqual('#', diff[2])
            self.assertEqual('#', diff[4])
            self.assertEqual('#', diff[6])
            self.assertEqual(1, diff.specificity)

    def test_should_get_differences_4(self):
        # Given
        p0 = Perception(['1', '1', '1', '1', '1', '0', '1', '0'])
        self.mark[0].update(['0', '1'])
        self.mark[1].update(['0', '1'])
        self.mark[3].update(['0', '1'])
        self.mark[4].update(['0', '1'])
        self.mark[6].update(['0', '1'])
        self.mark[7].update(['0'])

        # When
        diff = self.mark.get_differences(p0)

        # Then
        self.assertIsNotNone(diff)
        self.assertEqual(5, diff.specificity)
        self.assertEqual('1', diff[0])
        self.assertEqual('1', diff[1])
        self.assertEqual('#', diff[2])
        self.assertEqual('1', diff[3])
        self.assertEqual('1', diff[4])
        self.assertEqual('#', diff[5])
        self.assertEqual('1', diff[6])
        self.assertEqual('#', diff[7])

    def test_should_get_differences_5(self):
        # Given
        p0 = Perception(['0', '0', '2', '1', '1', '0', '1', '0'])
        self.mark[3] = '0'
        self.mark[6] = '0'

        for _ in range(100):
            # When
            diff = self.mark.get_differences(p0)

            # Then
            self.assertIsNotNone(diff)
            self.assertEqual(1, diff.specificity)

    def test_should_get_differences_6(self):
        # Given
        p0 = Perception(['1', '0', '1', '0', '1', '0', '0', '1'])

        # When
        diff = self.mark.get_differences(p0)

        # Then
        self.assertIsNone(diff)
