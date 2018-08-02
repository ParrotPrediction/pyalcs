import pytest

from lcs import Perception
from lcs.agents.acs2 import Effect


class TestEffect:

    def test_should_initialize_correctly(self):
        effect = Effect.empty(8)
        assert 8 == len(effect)

    def test_should_get_initialized_with_string(self):
        effect = Effect("#1O##O##")
        assert 8 == len(effect)

    def test_should_set_effect_with_non_string_char(self):
        effect = Effect.empty(8)

        with pytest.raises(TypeError):
            effect[0] = 1

    def test_should_return_number_of_specific_components_1(self):
        effect = Effect.empty(8)
        assert 0 == effect.number_of_specified_elements

    def test_should_return_number_of_specific_components_2(self):
        effect = Effect.empty(8)
        effect[1] = '1'

        assert 1 == effect.number_of_specified_elements

    def test_should_detect_correct_anticipation_1(self):
        # Classifier is not predicting any change, all pass-through effect
        # should predict correctly

        # given
        effect = Effect('########')
        p0 = Perception('00001111')
        p1 = Perception('00001111')

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is True

    def test_should_detect_correct_anticipation_2(self):
        # Introduce two changes into situation and effect (should
        # also predict correctly)

        # given
        effect = Effect(['#', '1', '#', '#', '#', '#', '0', '#'])
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '1', '0', '0', '1', '1', '0', '1'])

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is True

    def test_should_detect_correct_anticipation_3(self):
        # Case when effect predicts situation incorrectly

        # given
        effect = Effect(['#', '0', '#', '#', '#', '#', '#', '#'])
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '1', '0', '0', '1', '1', '1', '1'])

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is False

    def test_should_detect_correct_anticipation_4(self):
        # Case when effect predicts situation incorrectly

        # given
        effect = Effect(['#', '#', '0', '#', '#', '1', '#', '#'])
        p0 = Perception(['1', '0', '1', '0', '1', '0', '0', '1'])
        p1 = Perception(['1', '0', '1', '0', '1', '0', '0', '1'])

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is False

    def test_should_detect_correct_anticipation_5(self):
        # Case when effect predicts situation incorrectly

        # given
        effect = Effect(['#', '#', '#', '#', '1', '#', '0', '#'])
        p0 = Perception(['0', '1', '1', '0', '0', '0', '1', '1'])
        p1 = Perception(['1', '1', '1', '0', '1', '1', '0', '1'])

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is False

    def test_should_detect_correct_anticipation_6(self):
        # Case when effect predicts situation incorrectly

        # given
        effect = Effect(['#', '#', '1', '#', '0', '#', '0', '#'])
        p0 = Perception(['0', '0', '0', '1', '1', '0', '1', '0'])
        p1 = Perception(['0', '0', '1', '1', '0', '0', '0', '0'])

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is True

    def test_should_handle_pass_through_symbol(self):
        # A case when there was no change in perception but effect has no
        # pass-through symbol

        # given
        effect = Effect(['#', '0', '#', '#', '#', '#', '#', '#'])
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is False

    def test_should_check_if_specializable_1(self):
        # given
        p0 = Perception(['1', '1', '0', '0', '0', '0', '1', '0'])
        p1 = Perception(['1', '1', '1', '0', '1', '1', '0', '1'])
        effect = Effect(['#', '#', '#', '#', '#', '#', '#', '#'])

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is True

    def test_should_check_if_specializable_2(self):
        # given
        p0 = Perception(['0', '1', '1', '0', '0', '0', '1', '1'])
        p1 = Perception(['1', '1', '0', '0', '0', '0', '1', '0'])
        effect = Effect(['#', '#', '#', '#', '#', '#', '#', '#'])

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is True

    def test_should_check_if_specializable_3(self):
        # given
        p0 = Perception(['1', '1', '1', '1', '0', '1', '1', '1'])
        p1 = Perception(['1', '0', '0', '0', '0', '0', '0', '0'])
        effect = Effect(['#', '#', '0', '0', '#', '0', '#', '#'])

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is True

    def test_should_check_if_specializable_4(self):
        # given
        p0 = Perception(['1', '0', '0', '0', '0', '0', '0', '1'])
        p1 = Perception(['1', '0', '0', '0', '1', '0', '1', '1'])
        effect = Effect(['0', '#', '#', '#', '#', '1', '#', '#'])

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is False

    def test_should_check_if_specializable_5(self):
        # given
        p0 = Perception(['1', '1', '1', '1', '0', '1', '1', '1'])
        p1 = Perception(['1', '0', '1', '1', '1', '1', '1', '1'])
        effect = Effect(['#', '0', '1', '0', '#', '0', '1', '0'])

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is False

    def test_should_check_if_specializable_6(self):
        # given
        p0 = Perception(['0', '0', '1', '1', '0', '0', '0', '0'])
        p1 = Perception(['0', '0', '1', '1', '0', '0', '0', '0'])
        effect = Effect(['#', '1', '0', '#', '#', '#', '1', '1'])

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is False

    def test_should_check_if_specializable_7(self):
        # given
        p0 = Perception(['0', '1', '1', '0', '0', '0', '0', '0'])
        p1 = Perception(['1', '1', '0', '1', '1', '1', '0', '1'])
        effect = Effect(['#', '#', '0', '#', '#', '1', '#', '#'])

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is True

    def test_eq(self):
        assert Effect('00001111') == Effect('00001111')
        assert Effect('00001111') != Effect('0000111#')
