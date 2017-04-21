import unittest

from alcs.agent.acs3 import Effect


class EffectTest(unittest.TestCase):

    def setUp(self):
        self.effect = Effect()

    def test_should_initialize_correctly(self):
        self.assertTrue(len(self.effect) > 0)

    def test_should_return_number_of_specific_components(self):
        self.assertEqual(0, self.effect.number_of_specified_elements)

        self.effect[1] = '1'
        self.assertEqual(1, self.effect.number_of_specified_elements)
