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
        assert Effect(({"1": 0.6, "0": 0.4}, {"0": 0.8, "1": 0.2},
                       {"1": 1.0, "0": 0.0}, {"1": 1.0, "0": 0.0})) \
            == Effect(({"0": 0.4, "1": 0.6}, {"1": 0.2, "0": 0.8},
                       {"0": 0.0, "1": 1.0}, {"0": 0.0, "1": 1.0}))
        # Note that for probability-enhanced attributes, they are
        # considered "equal" when the algorithms are concerned, if
        # only they have the same symbols, no matter their probabilities
        assert Effect(({"1": 0.6, "0": 0.4}, {"0": 0.8, "1": 0.2},
                       {"1": 1.0, "0": 0.0}, {"1": 1.0, "0": 0.0})) \
            == Effect(({"0": 0.4, "1": 0.6}, {"1": 0.3, "0": 0.7},
                       {"0": 0.0, "1": 1.0}, {"0": 0.0, "1": 1.0}))

    @pytest.mark.parametrize("_p0, _p1, _e, _result", [
        (['1', '1', '0', '1', '1', '1', '0', '1'],
         ['1', '1', '1', '1', '1', '0', '0', '1'],
         ['#', '#', '0', '#', '#', '1', '#', '#'],
         True),
        (['1', '0', '1', '1', '1', '0', '0', '1'],
         ['1', '1', '0', '1', '1', '1', '0', '1'],
         ['#', '#', '0', '#', '#', '1', '#', '#'],
         False),
        (['1', '1', '1', '1', '0', '0', '0', '1'],
         ['1', '1', '0', '1', '1', '1', '0', '1'],
         ['#', '#', '0', '#', '#', '1', '#', '#'],
         False),
        (['1', '1', '1', '1', '1', '0', '1', '1'],
         ['1', '1', '1', '1', '1', '1', '0', '1'],
         ['#', '#', '0', '#', '#', '1', '#', '#'],
         False)
    ])
    def test_does_match(self, _p0, _p1, _e, _result):
        # given
        p0 = Perception(_p0)
        p1 = Perception(_p1)
        effect = Effect(_e)

        # when
        result = effect.does_match(p0, p1)

        # then
        assert result is _result

    @pytest.mark.parametrize("_p0, _result", [
        (['1', '1', '0', '1', '1', '1', '1', '1'],
         ['1', '1', '0', '1', '1', '0', '1', '1']),
        (['1', '1', '1', '1', '1', '1', '1', '1'],
         ['1', '1', '0', '1', '1', '0', '1', '1'])
    ])
    def test_get_best_anticipation(self, _p0, _result):
        # given
        p0 = Perception(_p0)
        effect = Effect(['#', '#', '0', '#', '#', '0', '#', '#'])

        # when
        result0 = effect.get_best_anticipation(p0)

        # then
        assert result0 == _result

    @pytest.mark.parametrize("_p0, _p1, _e, _result", [
        (['1', '1', '0', '1', '1', '1', '1', '0'],
         ['1', '1', '1', '1', '1', '1', '1', '0'],
         ['#', '#', '1', '#', '#', '0', '#', '#'],
         True),
        (['1', '1', '0', '1', '1', '1', '1', '0'],
         ['1', '1', '1', '1', '1', '1', '0', '0'],
         ['#', '#', '1', '#', '#', '0', '#', '#'],
         False),
        (['1', '1', '0', '1', '1', '0', '1', '0'],
         ['1', '1', '1', '1', '1', '1', '1', '0'],
         ['#', '#', '1', '#', '#', '0', '#', '#'],
         False)
    ])
    def test_does_specify_only_changes_backwards(self, _p0, _p1, _e, _result):
        # given
        back_ant = Perception(_p0)
        sit = Perception(_p1)
        effect = Effect(_e)

        # when
        result = effect.does_specify_only_changes_backwards(back_ant, sit)

        # then
        assert result is _result

    def test_str(self):
        # given
        effect = Effect(({"1": 0.6, "0": 0.4}, {"0": 0.8, "1": 0.2}, "1", "1"))

        # then
        assert str(effect) == "{10}{01}11"

    def test_reduced_to_non_enhanced(self):
        # given
        effect = Effect(({"1": 0.6, "0": 0.4}, {"0": 0.8, "1": 0.2}, "1", "1"))

        # then
        assert effect.reduced_to_non_enhanced() == Effect("1011")

    def test_not_enhanced(self):
        # given
        effect = Effect(("1", "0", "1", "1"))

        # then
        assert not effect.is_enhanced()

    def test_enhanced(self):
        # given
        effect = Effect(({"1": 0.6, "0": 0.4}, {"0": 0.8, "1": 0.2}, "1", "1"))

        # then
        assert effect.is_enhanced()

    def test_update_equivalence(self):
        # given (note that the effects are practically equivalent)
        perception = Perception("1101")
        effect_a = Effect(
            ({"1": 0.6, "0": 0.4}, {"0": 0.8, "1": 0.2}, "1", "1"))
        effect_b = Effect(({"1": 0.6, "0": 0.4}, {"0": 0.8, "1": 0.2},
                           {"1": 1.0, "0": 0.0}, {"1": 1.0, "0": 0.0}))

        # when
        effect_a.update_enhanced_effect_probs(perception, 0.6)
        effect_b.update_enhanced_effect_probs(perception, 0.6)

        # then
        assert effect_a == effect_b
