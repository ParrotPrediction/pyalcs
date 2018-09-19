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

    @pytest.mark.parametrize("_e, _result", [
        ("########", False),
        ("#######1", True),
        ("11111111", True),
    ])
    def test_should_detect_change(self, _e, _result):
        assert Effect(_e).specify_change == _result

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

    def test_does_match(self):
        # given
        p0 = Perception(['1', '1', '0', '1', '1', '1', '0', '1'])
        p1 = Perception(['1', '1', '1', '1', '1', '0', '0', '1'])
        effect = Effect(['#', '#', '0', '#', '#', '1', '#', '#'])

        # when
        result = effect.does_match(p0, p1)

        # then
        assert result is True

    def test_does_match_false(self):
        # given
        p0 = Perception(['1', '0', '1', '1', '1', '0', '0', '1'])
        p1 = Perception(['1', '1', '0', '1', '1', '1', '0', '1'])
        effect = Effect(['#', '#', '0', '#', '#', '1', '#', '#'])

        # when
        result = effect.does_match(p0, p1)

        # then
        assert result is False

    def test_does_match_false_2(self):
        # given
        p0 = Perception(['1', '1', '1', '1', '0', '0', '0', '1'])
        p1 = Perception(['1', '1', '0', '1', '1', '1', '0', '1'])
        effect = Effect(['#', '#', '0', '#', '#', '1', '#', '#'])

        # when
        result = effect.does_match(p0, p1)

        # then
        assert result is False

    def test_does_match_false_3(self):
        # given
        p0 = Perception(['1', '1', '1', '1', '1', '0', '1', '1'])
        p1 = Perception(['1', '1', '1', '1', '1', '1', '0', '1'])
        effect = Effect(['#', '#', '0', '#', '#', '1', '#', '#'])

        # when
        result = effect.does_match(p0, p1)

        # then
        assert result is False

    def test_get_best_anticipation(self):
        # given
        p0 = Perception(['1', '1', '0', '1', '1', '1', '1', '1'])
        p1 = Perception(['1', '1', '1', '1', '1', '1', '1', '1'])
        effect = Effect(['#', '#', '0', '#', '#', '0', '#', '#'])

        # when
        result0 = effect.get_best_anticipation(p0)
        result1 = effect.get_best_anticipation(p1)

        # then
        assert result0 == ['1', '1', '0', '1', '1', '0', '1', '1']
        assert result1 == ['1', '1', '0', '1', '1', '0', '1', '1']

    def test_does_specify_only_changes_backwards(self):
        # given
        back_ant = Perception(['1', '1', '0', '1', '1', '1', '1', '0'])
        sit = Perception(['1', '1', '1', '1', '1', '1', '1', '0'])
        effect = Effect(['#', '#', '1', '#', '#', '0', '#', '#'])

        # when
        result = effect.does_specify_only_changes_backwards(back_ant, sit)

        # then
        assert result is True

    def test_does_specify_only_changes_backwards_false(self):
        # given
        back_ant = Perception(['1', '1', '0', '1', '1', '1', '1', '0'])
        sit = Perception(['1', '1', '1', '1', '1', '1', '0', '0'])
        effect = Effect(['#', '#', '1', '#', '#', '0', '#', '#'])

        # when
        result = effect.does_specify_only_changes_backwards(back_ant, sit)

        # then
        assert result is False

    def test_does_specify_only_changes_backwards_false_2(self):
        # given
        back_ant = Perception(['1', '1', '0', '1', '1', '0', '1', '0'])
        sit = Perception(['1', '1', '1', '1', '1', '1', '1', '0'])
        effect = Effect(['#', '#', '1', '#', '#', '0', '#', '#'])

        # when
        result = effect.does_specify_only_changes_backwards(back_ant, sit)

        # then
        assert result is False
