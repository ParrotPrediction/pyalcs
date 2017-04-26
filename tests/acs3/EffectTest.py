import unittest

from alcs.agent import Perception
from alcs.agent.acs3 import Condition, Effect


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

    def test_should_get_and_specialize_no_changes(self):
        e = Effect()
        c = e.get_and_specialize(self.previous_situation, self.situation)
        self.assertEqual(Effect(['#', '#', '#', '#', '#', '#', '#', '#']), e)
        self.assertEqual(Condition(['#', '#', '#', '#', '#', '#', '#', '#']), c)

    def test_should_get_and_specialize_changes(self):
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '0', '0', '1', '1', '1', '1', '1'])

        e = Effect()
        c = e.get_and_specialize(p0, p1)
        self.assertEqual(Effect(['#', '#', '#', '1', '#', '#', '#', '#']), e)
        self.assertEqual(Condition(['#', '#', '#', '0', '#', '#', '#', '#']), c)
