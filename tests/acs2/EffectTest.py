import unittest

from alcs.acs2 import Effect
from alcs import Perception


class EffectTest(unittest.TestCase):

    def setUp(self):
        self.effect = Effect()

    def test_should_initialize_correctly(self):
        self.assertTrue(len(self.effect) > 0)

    def test_should_get_initialized_with_str_1(self):
        eff = "#1O##O##"
        self.effect = Effect(eff)
        self.assertEqual(8, len(self.effect))

    def test_should_get_initialized_with_str_2(self):
        eff = "#1O##O#"
        # Too short effect
        self.assertRaises(ValueError, Effect, eff)

    def test_should_set_effect_with_non_string_char(self):
        self.assertRaises(TypeError, self.effect.__setitem__, 0, 1)

    def test_should_return_number_of_specific_components_1(self):
        self.assertEqual(0, self.effect.number_of_specified_elements)

    def test_should_return_number_of_specific_components_2(self):
        self.effect[1] = '1'
        self.assertEqual(1, self.effect.number_of_specified_elements)

    def test_should_detect_correct_anticipation_1(self):
        # Classifier is not predicting any change, all pass-through effect
        # should predict correctly

        # Given
        self.effect = Effect('########')
        p0 = Perception('00001111')
        p1 = Perception('00001111')

        # When
        res = self.effect.does_anticipate_correctly(p0, p1)

        # Then
        self.assertTrue(res)

    def test_should_detect_correct_anticipation_2(self):
        # Introduce two changes into situation and effect (should
        # also predict correctly)

        # Given
        self.effect = Effect(['#', '1', '#', '#', '#', '#', '0', '#'])
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '1', '0', '0', '1', '1', '0', '1'])

        # When
        res = self.effect.does_anticipate_correctly(p0, p1)

        # Then
        self.assertTrue(res)

    def test_should_detect_correct_anticipation_3(self):
        # Case when effect predicts situation incorrectly

        # Given
        self.effect = Effect(['#', '0', '#', '#', '#', '#', '#', '#'])
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '1', '0', '0', '1', '1', '1', '1'])

        # When
        res = self.effect.does_anticipate_correctly(p0, p1)

        # Then
        self.assertFalse(res)

    def test_should_detect_correct_anticipation_4(self):
        # Case when effect predicts situation incorrectly

        # Given
        self.effect = Effect(['#', '#', '0', '#', '#', '1', '#', '#'])
        p0 = Perception(['1', '0', '1', '0', '1', '0', '0', '1'])
        p1 = Perception(['1', '0', '1', '0', '1', '0', '0', '1'])

        # When
        res = self.effect.does_anticipate_correctly(p0, p1)

        # Then
        self.assertFalse(res)

    def test_should_detect_correct_anticipation_5(self):
        # Case when effect predicts situation incorrectly

        # Given
        self.effect = Effect(['#', '#', '#', '#', '1', '#', '0', '#'])
        p0 = Perception(['0', '1', '1', '0', '0', '0', '1', '1'])
        p1 = Perception(['1', '1', '1', '0', '1', '1', '0', '1'])

        # When
        res = self.effect.does_anticipate_correctly(p0, p1)

        # Then
        self.assertFalse(res)

    def test_should_detect_correct_anticipation_6(self):
        # Case when effect predicts situation incorrectly

        # Given
        self.effect = Effect(['#', '#', '1', '#', '0', '#', '0', '#'])
        p0 = Perception(['0', '0', '0', '1', '1', '0', '1', '0'])
        p1 = Perception(['0', '0', '1', '1', '0', '0', '0', '0'])

        # When
        res = self.effect.does_anticipate_correctly(p0, p1)

        # Then
        self.assertTrue(res)

    def test_should_handle_pass_through_symbol(self):
        # A case when there was no change in perception but effect has no
        # pass-through symbol

        # Given
        self.effect = Effect(['#', '0', '#', '#', '#', '#', '#', '#'])
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])

        # When
        res = self.effect.does_anticipate_correctly(p0, p1)

        # Then
        self.assertFalse(res)

    def test_should_check_if_specializable_1(self):
        # Given
        p0 = Perception(['1', '1', '0', '0', '0', '0', '1', '0'])
        p1 = Perception(['1', '1', '1', '0', '1', '1', '0', '1'])
        e =      Effect(['#', '#', '#', '#', '#', '#', '#', '#'])

        # When
        res = e.is_specializable(p0, p1)

        # Then
        self.assertTrue(res)

    def test_should_check_if_specializable_2(self):
        # Given
        p0 = Perception(['0', '1', '1', '0', '0', '0', '1', '1'])
        p1 = Perception(['1', '1', '0', '0', '0', '0', '1', '0'])
        e =      Effect(['#', '#', '#', '#', '#', '#', '#', '#'])

        # When
        res = e.is_specializable(p0, p1)

        # Then
        self.assertTrue(res)

    def test_should_check_if_specializable_3(self):
        # Given
        p0 = Perception(['1', '1', '1', '1', '0', '1', '1', '1'])
        p1 = Perception(['1', '0', '0', '0', '0', '0', '0', '0'])
        e =      Effect(['#', '#', '0', '0', '#', '0', '#', '#'])

        # When
        res = e.is_specializable(p0, p1)

        # Then
        self.assertTrue(res)

    def test_should_check_if_specializable_4(self):
        # Given
        p0 = Perception(['1', '0', '0', '0', '0', '0', '0', '1'])
        p1 = Perception(['1', '0', '0', '0', '1', '0', '1', '1'])
        e =      Effect(['0', '#', '#', '#', '#', '1', '#', '#'])

        # When
        res = e.is_specializable(p0, p1)

        # Then
        self.assertFalse(res)

    def test_should_check_if_specializable_5(self):
        # Given
        p0 = Perception(['1', '1', '1', '1', '0', '1', '1', '1'])
        p1 = Perception(['1', '0', '1', '1', '1', '1', '1', '1'])
        e =      Effect(['#', '0', '1', '0', '#', '0', '1', '0'])

        # When
        res = e.is_specializable(p0, p1)

        # Then
        self.assertFalse(res)

    def test_should_check_if_specializable_6(self):
        # Given
        p0 = Perception(['0', '0', '1', '1', '0', '0', '0', '0'])
        p1 = Perception(['0', '0', '1', '1', '0', '0', '0', '0'])
        e =      Effect(['#', '1', '0', '#', '#', '#', '1', '1'])

        # When
        res = e.is_specializable(p0, p1)

        # Then
        self.assertFalse(res)

    def test_should_check_if_specializable_7(self):
        # Given
        p0 = Perception(['0', '1', '1', '0', '0', '0', '0', '0'])
        p1 = Perception(['1', '1', '0', '1', '1', '1', '0', '1'])
        e =      Effect(['#', '#', '0', '#', '#', '1', '#', '#'])

        # When
        res = e.is_specializable(p0, p1)

        # Then
        self.assertTrue(res)

    def testEq(self):
        self.assertTrue(Effect('00001111') == Effect('00001111'))
