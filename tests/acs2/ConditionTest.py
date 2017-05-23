import unittest

from alcs.agent.Perception import Perception
from alcs.agent.acs2 import Condition


class ConditionTest(unittest.TestCase):
    def setUp(self):
        self.c = Condition()

    def test_should_only_accept_strings(self):
        # Try to store an integer
        self.assertRaises(TypeError, self.c.__setitem__, 0, 1)

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
        c =    Condition()
        diff =   Condition(['#', '0', '#', '#', '#', '1', '#', '1'])
        result = Condition(['#', '0', '#', '#', '#', '1', '#', '1'])
        c.specialize(new_condition=diff)
        self.assertEqual(result, c)

    def test_should_specialize_2(self):
        c =      Condition(['#', '#', '#', '1', '0', '#', '1', '#'])
        diff =   Condition(['0', '1', '0', '#', '#', '1', '#', '#'])
        result = Condition(['0', '1', '0', '1', '0', '1', '1', '#'])
        c.specialize(new_condition=diff)
        self.assertEqual(result, c)

    def test_should_specialize_3(self):
        c =      Condition(['#', '1', '0', '1', '#', '1', '0', '#'])
        diff =   Condition(['#', '#', '#', '#', '1', '#', '#', '1'])
        result = Condition(['#', '1', '0', '1', '1', '1', '0', '1'])
        c.specialize(new_condition=diff)
        self.assertEqual(result, c)

    def test_should_specialize_4(self):
        c =      Condition(['#', '#', '#', '#', '0', '1', '#', '1'])
        diff =   Condition(['2', '#', '0', '0', '#', '#', '#', '#'])
        result = Condition(['2', '#', '0', '0', '0', '1', '#', '1'])
        c.specialize(new_condition=diff)
        self.assertEqual(result, c)

    def test_should_specialize_5(self):
        c =      Condition(['#', '#', '#', '0', '1', '#', '0', '#'])
        diff =   Condition(['1', '0', '1', '#', '#', '0', '#', '#'])
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
        self.assertRaises(ValueError, self.c.does_match, Perception(['1', '2']))

    def test_should_match_condition(self):
        c = Condition(['1', '0', '0', '1', '1', '0', '0', '1'])

        # General condition - should match everything
        self.assertTrue(self.c.does_match(c))

        # Correct first position
        self.c.specialize(0, '1')
        self.assertTrue(self.c.does_match(c))

        # Expects 0 as the first condition
        self.c.specialize(0, '0')
        self.assertFalse(self.c.does_match(c))

