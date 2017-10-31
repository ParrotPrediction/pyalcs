import unittest

from alcs.acs2 import Condition

from alcs import Perception
from alcs.acs2.testrandom import TestSample


class ConditionTest(unittest.TestCase):
    def setUp(self):
        self.c = Condition()

    def test_equal(self):  # TODO remove
        self.assertTrue(Condition('########').equal(Condition('########')))
        self.assertFalse(Condition('1#######').equal(Condition('########')))
        self.assertFalse(Condition('########').equal(Condition('#######1')))
        self.assertTrue(Condition('1111####').equal(Condition('1111####')))
        self.assertFalse(Condition('1111####').equal(Condition('1011####')))
        self.assertFalse(Condition('1101####').equal(Condition('1111####')))
        self.assertTrue(Condition('00001###').equal(Condition('00001###')))

    def test_should_generalize(self):
        cond = "#1O##O##"
        self.c = Condition(cond)
        self.c.generalize(position=1)
        expected_result = Condition("##O##O##")
        self.assertEqual(expected_result, self.c)

    def test_generalize_decrements_specificity(self):
        self.c = Condition('#11#####')
        self.assertEqual(2, self.c.specificity)
        self.c.generalize(1)
        self.assertEqual(1, self.c.specificity)

    def test_should_only_accept_strings(self):
        # Try to store an integer
        self.assertRaises(TypeError, self.c.__setitem__, 0, 1)

    def test_should_initialize_two_times_the_same_way(self):
        c1 = Condition("#1O##O##")
        c2 = Condition("#1O##O##")
        self.assertEqual(c1, c2)

    def test_should_return_number_of_specified_elements(self):
        self.assertEqual(0, self.c.specificity)

        self.c.specialize(2, '1')
        self.c.specialize(5, '0')
        self.assertEqual(2, self.c.specificity)

    def test_should_get_initialized_with_str_1(self):
        cond = "#1O##O##"
        self.c = Condition(cond)
        self.assertEqual(8, len(self.c))

    def test_should_get_initialized_with_str_2(self):
        cond = "#1O##O#"
        # Too short condition
        self.assertRaises(ValueError, Condition, cond)

    def test_should_specialize_1(self):
        c = Condition()
        diff = Condition(['#', '0', '#', '#', '#', '1', '#', '1'])
        result = Condition(['#', '0', '#', '#', '#', '1', '#', '1'])
        c.specialize(new_condition=diff)
        self.assertEqual(result, c)

    def test_should_specialize_2(self):
        c = Condition(['#', '#', '#', '1', '0', '#', '1', '#'])
        diff = Condition(['0', '1', '0', '#', '#', '1', '#', '#'])
        result = Condition(['0', '1', '0', '1', '0', '1', '1', '#'])
        c.specialize(new_condition=diff)
        self.assertEqual(result, c)

    def test_should_specialize_3(self):
        c = Condition(['#', '1', '0', '1', '#', '1', '0', '#'])
        diff = Condition(['#', '#', '#', '#', '1', '#', '#', '1'])
        result = Condition(['#', '1', '0', '1', '1', '1', '0', '1'])
        c.specialize(new_condition=diff)
        self.assertEqual(result, c)

    def test_should_specialize_4(self):
        c = Condition(['#', '#', '#', '#', '0', '1', '#', '1'])
        diff = Condition(['2', '#', '0', '0', '#', '#', '#', '#'])
        result = Condition(['2', '#', '0', '0', '0', '1', '#', '1'])
        c.specialize(new_condition=diff)
        self.assertEqual(result, c)

    def test_should_specialize_5(self):
        c = Condition(['#', '#', '#', '0', '1', '#', '0', '#'])
        diff = Condition(['1', '0', '1', '#', '#', '0', '#', '#'])
        result = Condition(['1', '0', '1', '0', '1', '0', '0', '#'])
        c.specialize(new_condition=diff)
        self.assertEqual(result, c)

    def test_should_match_perception(self):
        p = Perception(['1', '0', '0', '1', '1', '0', '0', '1'])

        # General condition - should match everything
        self.assertTrue(self.c.does_match(p))

        # Correct first position
        self.c.specialize(0, '1')
        self.assertTrue(self.c.does_match(p))

        # Expects 0 as the first condition
        self.c.specialize(0, '0')
        self.assertFalse(self.c.does_match(p))

        # Should fail when perception length is different
        self.assertRaises(ValueError,
                          self.c.does_match,
                          Perception(['1', '2']))

    def test_should_match_condition_1(self):
        c = Condition(['1', '0', '0', '1', '1', '0', '0', '1'])

        # General condition - should match everything
        self.assertTrue(self.c.does_match(c))

        # Correct first position
        self.c.specialize(0, '1')
        self.assertTrue(self.c.does_match(c))

        # Expects 0 as the first condition
        self.c.specialize(0, '0')
        self.assertFalse(self.c.does_match(c))

    def test_should_match_condition_2(self):
        # Given
        self.c = Condition('####O###')
        other = Condition('#1O##O##')

        # When
        res = self.c.does_match(other)

        # Then
        self.assertTrue(res)

    def test_crossover(self):
        c1 = Condition('0##10###')
        c2 = Condition('#10##0##')
        c1.two_point_crossover(c2, samplefunc=TestSample([1, 4]))
        self.assertEqual(Condition('010#0###'), c1)
        self.assertEqual(Condition('###1#0##'), c2)

    def test_crossover_allows_to_change_last_element(self):
        c1 = Condition('0##10###')
        c2 = Condition('#10##011')
        c1.two_point_crossover(c2, samplefunc=TestSample([5, 8]))
        self.assertEqual(Condition('0##10011'), c1)
        self.assertEqual(Condition('#10#####'), c2)
