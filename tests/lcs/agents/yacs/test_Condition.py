import pytest

from lcs import Perception
from lcs.agents.yacs.yacs import Condition, DontCare


class TestCondition:

    def test_should_get_initialized(self):
        # given
        condition = Condition("#1O#")

        # then
        assert len(condition) == 4
        assert type(condition[0]) is DontCare
        assert type(condition[1]) is str
        assert type(condition[2]) is str
        assert type(condition[3]) is DontCare
        assert condition[0].eis == 0.0
        assert condition[3].eis == 0.0

    def test_should_generate_condition(self):
        # given
        p0 = Perception('1010')

        # when
        c = next(Condition.random_matching(p0))

        # then
        assert c is not None
        assert type(c) is Condition

    @pytest.mark.parametrize('_self, _other, _res', [
        ('####', '####', False),
        ('####', '###1', True),
        ('###1', '###1', False),
        ('###1', '##11', True),
        ('2##1', '1#11', False),
    ])
    def test_should_determine_more_specified(self, _self, _other, _res):
        assert Condition(_self).is_more_specialized(Condition(_other)) == _res

    @pytest.mark.parametrize('_self, _other, _res', [
        ('####', '####', False),
        ('####', '###1', False),
        ('###1', '###1', False),
        ('##11', '###1', True),
        ('##11', '####', True),
        ('1#11', '###2', False),
        ('1#11', '###1', True),
    ])
    def test_should_determine_more_general(self, _self, _other, _res):
        assert Condition(_self).is_more_general(Condition(_other)) == _res

    @pytest.mark.parametrize('_c, _p, _res', [
        ("#000", "0000", True),
        ("#000", "1000", True),
        ("#000", "1001", False),
    ])
    def test_should_match_perception(self, _c, _p, _res):
        assert Condition(_c).does_match(Perception(_p)) is _res

    @pytest.mark.parametrize('_c, _res', [
        ("####", [0.0, 0.0, 0.0, 0.0]),
        ("##1#", [0.0, 0.0, 0.0, 0.0]),
    ])
    def test_should_get_expected_improvements_property(self, _c, _res):
        assert Condition(_c).expected_improvements == _res

    @pytest.mark.parametrize('_c, _gen, _spec', [
        ('####', 4, 0),
        ('###1', 3, 1),
        ('1111', 0, 4),
    ])
    def test_should_calculate_generality_specificity(self, _c, _gen, _spec):
        condition = Condition(_c)
        assert condition.generality == _gen
        assert condition.specificity == _spec
