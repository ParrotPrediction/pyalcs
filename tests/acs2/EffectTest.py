import unittest

from alcs.agent import Perception
from alcs.agent.acs2 import Effect


class EffectTest(unittest.TestCase):

    def setUp(self):
        self.effect = Effect()
        self.previous_situation = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        self.situation = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])

    def test_should_initialize_correctly(self):
        self.assertTrue(len(self.effect) > 0)

    def test_should_return_number_of_specific_components(self):
        self.assertEqual(0, self.effect.number_of_specified_elements)

        self.effect[1] = '1'
        self.assertEqual(1, self.effect.number_of_specified_elements)

    def test_should_anticipate_correctly_no_change(self):
        # Classifier is not predicting any change, all pass-through effect
        # should predict correctly
        self.effect = Effect(['#', '#', '#', '#', '#', '#', '#', '#'])
        self.assertTrue(self.effect.does_anticipate_correctly(
            self.previous_situation, self.situation))

    def test_should_anticipate_changed_attributes(self):
        # Introduce two changes into situation and effect (should
        # also predict correctly)
        self.effect = Effect(['#', '1', '#', '#', '#', '#', '0', '#'])
        self.situation[1] = '1'
        self.situation[6] = '0'

        self.assertTrue(self.effect.does_anticipate_correctly(
            self.previous_situation, self.situation))

    def test_should_handle_wrong_anticipation(self):
        # Case when effect predicts situation incorrectly
        self.effect = Effect(['#', '0', '#', '#', '#', '#', '#', '#'])
        self.situation[1] = '1'

        self.assertFalse(self.effect.does_anticipate_correctly(
            self.previous_situation, self.situation
        ))

    def test_should_handle_pass_through_symbol(self):
        # A case when there was no change in perception but effect has no
        # pass-through symbol
        self.effect = Effect(['#', '0', '#', '#', '#', '#', '#', '#'])
        self.assertFalse(self.effect.does_anticipate_correctly(
            self.previous_situation, self.situation
        ))

    def test_should_check_if_specializable(self):
        p0 = Perception(['1', '1', '0', '0', '0', '0', '1', '0'])
        p1 = Perception(['1', '1', '1', '0', '1', '1', '0', '1'])
        e = Effect(     ['#', '#', '#', '#', '#', '#', '#', '#'])
        self.assertTrue(e.is_specializable(p0, p1))

        p0 = Perception(['0', '1', '1', '0', '0', '0', '1', '1'])
        p1 = Perception(['1', '1', '0', '0', '0', '0', '1', '0'])
        e = Effect(     ['#', '#', '#', '#', '#', '#', '#', '#'])
        self.assertTrue(e.is_specializable(p0, p1))

        p0 = Perception(['1', '1', '1', '1', '0', '1', '1', '1'])
        p1 = Perception(['1', '0', '0', '0', '0', '0', '0', '0'])
        e = Effect(     ['#', '#', '0', '0', '#', '0', '#', '#'])
        self.assertTrue(e.is_specializable(p0, p1))

        p0 = Perception(['1', '0', '0', '0', '0', '0', '0', '1'])
        p1 = Perception(['1', '0', '0', '0', '1', '0', '1', '1'])
        e = Effect(     ['0', '#', '#', '#', '#', '1', '#', '#'])
        self.assertFalse(e.is_specializable(p0, p1))

        p0 = Perception(['1', '1', '1', '1', '0', '1', '1', '1'])
        p1 = Perception(['1', '0', '1', '1', '1', '1', '1', '1'])
        e = Effect(     ['#', '0', '1', '0', '#', '0', '1', '0'])
        self.assertFalse(e.is_specializable(p0, p1))
