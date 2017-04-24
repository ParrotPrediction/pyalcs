import unittest

from alcs.agent.Perception import Perception
from alcs.agent.acs3 import Condition


class ConditionTest(unittest.TestCase):

    def setUp(self):
        self.c = Condition()

    def test_should_only_accept_strings(self):
        # Try to store an integer
        self.assertRaises(TypeError, self.c.__setitem__, 0, 1)

    def test_should_return_number_of_specified_elements(self):
        self.assertEqual(0, self.c.number_of_specified_elements)

        self.c.specialize(2, '1')
        self.c.specialize(5, '0')
        self.assertEqual(2, self.c.number_of_specified_elements)

    def test_should_match_perception(self):
        p = Perception(['1', '0', '0', '1', '1', '0', '0', '1'])

        # General perception - should match everything
        self.assertTrue(self.c.does_match(p))

        # Correct first position
        self.c.specialize(0, '1')
        self.assertTrue(self.c.does_match(p))

        # Expects 0 as the first condition
        self.c.specialize(0, '0')
        self.assertFalse(self.c.does_match(p))

        # Should fail when perception length is different
        self.assertRaises(ValueError, self.c.does_match, Perception(['1', '2']))