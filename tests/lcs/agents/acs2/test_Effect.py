import pytest

from lcs import Perception
from lcs.agents.acs2 import ACS2Configuration, Effect


class TestEffect:

    @pytest.fixture
    def cfg(self):
        return ACS2Configuration(8, 8)

    def test_should_initialize_correctly(self, cfg):
        effect = Effect(cfg=cfg)
        assert len(effect) > 0

    def test_should_get_initialized_with_str_1(self, cfg):
        effect = Effect("#1O##O##", cfg)
        assert 8 == len(effect)

    def test_should_get_initialized_with_str_2(self, cfg):
        with pytest.raises(ValueError):
            # Too short effect
            Effect("#1O##O#", cfg)

    def test_should_set_effect_with_non_string_char(self, cfg):
        effect = Effect(cfg=cfg)

        with pytest.raises(TypeError):
            effect[0] = 1

    def test_should_return_number_of_specific_components_1(self, cfg):
        effect = Effect(cfg=cfg)
        assert 0 == effect.number_of_specified_elements

    def test_should_return_number_of_specific_components_2(self, cfg):
        effect = Effect(cfg=cfg)
        effect[1] = '1'

        assert 1 == effect.number_of_specified_elements

    def test_should_detect_correct_anticipation_1(self, cfg):
        # Classifier is not predicting any change, all pass-through effect
        # should predict correctly

        # given
        effect = Effect('########', cfg)
        p0 = Perception('00001111')
        p1 = Perception('00001111')

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # ghen
        assert res is True

    def test_should_detect_correct_anticipation_2(self, cfg):
        # Introduce two changes into situation and effect (should
        # also predict correctly)

        # given
        effect = Effect(['#', '1', '#', '#', '#', '#', '0', '#'], cfg)
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '1', '0', '0', '1', '1', '0', '1'])

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is True

    def test_should_detect_correct_anticipation_3(self, cfg):
        # Case when effect predicts situation incorrectly

        # given
        effect = Effect(['#', '0', '#', '#', '#', '#', '#', '#'], cfg)
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '1', '0', '0', '1', '1', '1', '1'])

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is False

    def test_should_detect_correct_anticipation_4(self, cfg):
        # Case when effect predicts situation incorrectly

        # given
        effect = Effect(['#', '#', '0', '#', '#', '1', '#', '#'], cfg)
        p0 = Perception(['1', '0', '1', '0', '1', '0', '0', '1'])
        p1 = Perception(['1', '0', '1', '0', '1', '0', '0', '1'])

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is False

    def test_should_detect_correct_anticipation_5(self, cfg):
        # Case when effect predicts situation incorrectly

        # given
        effect = Effect(['#', '#', '#', '#', '1', '#', '0', '#'], cfg)
        p0 = Perception(['0', '1', '1', '0', '0', '0', '1', '1'])
        p1 = Perception(['1', '1', '1', '0', '1', '1', '0', '1'])

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is False

    def test_should_detect_correct_anticipation_6(self, cfg):
        # Case when effect predicts situation incorrectly

        # given
        effect = Effect(['#', '#', '1', '#', '0', '#', '0', '#'], cfg)
        p0 = Perception(['0', '0', '0', '1', '1', '0', '1', '0'])
        p1 = Perception(['0', '0', '1', '1', '0', '0', '0', '0'])

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is True

    def test_should_handle_pass_through_symbol(self, cfg):
        # A case when there was no change in perception but effect has no
        # pass-through symbol

        # given
        effect = Effect(['#', '0', '#', '#', '#', '#', '#', '#'], cfg)
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])

        # when
        res = effect.does_anticipate_correctly(p0, p1)

        # then
        assert res is False

    def test_should_check_if_specializable_1(self, cfg):
        # given
        p0 = Perception(['1', '1', '0', '0', '0', '0', '1', '0'])
        p1 = Perception(['1', '1', '1', '0', '1', '1', '0', '1'])
        effect = Effect(['#', '#', '#', '#', '#', '#', '#', '#'], cfg)

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is True

    def test_should_check_if_specializable_2(self, cfg):
        # given
        p0 = Perception(['0', '1', '1', '0', '0', '0', '1', '1'])
        p1 = Perception(['1', '1', '0', '0', '0', '0', '1', '0'])
        effect = Effect(['#', '#', '#', '#', '#', '#', '#', '#'], cfg)

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is True

    def test_should_check_if_specializable_3(self, cfg):
        # given
        p0 = Perception(['1', '1', '1', '1', '0', '1', '1', '1'])
        p1 = Perception(['1', '0', '0', '0', '0', '0', '0', '0'])
        effect = Effect(['#', '#', '0', '0', '#', '0', '#', '#'], cfg)

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is True

    def test_should_check_if_specializable_4(self, cfg):
        # given
        p0 = Perception(['1', '0', '0', '0', '0', '0', '0', '1'])
        p1 = Perception(['1', '0', '0', '0', '1', '0', '1', '1'])
        effect = Effect(['0', '#', '#', '#', '#', '1', '#', '#'], cfg)

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is False

    def test_should_check_if_specializable_5(self, cfg):
        # given
        p0 = Perception(['1', '1', '1', '1', '0', '1', '1', '1'])
        p1 = Perception(['1', '0', '1', '1', '1', '1', '1', '1'])
        effect = Effect(['#', '0', '1', '0', '#', '0', '1', '0'], cfg)

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is False

    def test_should_check_if_specializable_6(self, cfg):
        # given
        p0 = Perception(['0', '0', '1', '1', '0', '0', '0', '0'])
        p1 = Perception(['0', '0', '1', '1', '0', '0', '0', '0'])
        effect = Effect(['#', '1', '0', '#', '#', '#', '1', '1'], cfg)

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is False

    def test_should_check_if_specializable_7(self, cfg):
        # given
        p0 = Perception(['0', '1', '1', '0', '0', '0', '0', '0'])
        p1 = Perception(['1', '1', '0', '1', '1', '1', '0', '1'])
        effect = Effect(['#', '#', '0', '#', '#', '1', '#', '#'], cfg)

        # when
        res = effect.is_specializable(p0, p1)

        # then
        assert res is True

    def test_eq(self, cfg):
        assert Effect('00001111', cfg) == Effect('00001111', cfg)
        assert Effect('00001111', cfg) != Effect('0000111#', cfg)
