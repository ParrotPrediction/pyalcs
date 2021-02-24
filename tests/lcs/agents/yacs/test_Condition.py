import random

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
        ('2##1', '1#11', False),  # different token on first position
        ('2##1', '1###', False),
    ])
    def test_should_determine_more_specialized(self, _self, _other, _res):
        # given
        cond = Condition(_self)
        other = Condition(_other)

        # randomly insert some eis values
        for t in cond:
            if type(t) is DontCare and random.random() < 0.5:
                t.eis = random.random()

        for t in other:
            if type(t) is DontCare and random.random() < 0.5:
                t.eis = random.random()

        # then
        assert cond.is_more_specialized(other) == _res

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
        # given
        cond = Condition(_self)
        other = Condition(_other)

        # randomly insert some eis values
        for t in cond:
            if type(t) is DontCare and random.random() < 0.5:
                t.eis = random.random()

        for t in other:
            if type(t) is DontCare and random.random() < 0.5:
                t.eis = random.random()

        # then
        assert cond.is_more_general(other) == _res

    @pytest.mark.parametrize('_c, _p, _res', [
        ("#000", "0000", True),
        ("#000", "1000", True),
        ("#000", "1001", False),
    ])
    def test_should_match_perception(self, _c, _p, _res):
        assert Condition(_c).does_match(Perception(_p)) is _res

    def test_should_match_perception_with_eis(self,):
        # given
        cond = Condition('#000')
        cond[0].eis = 0.2
        p0 = Perception('0000')

        assert cond.does_match(p0)

    @pytest.mark.parametrize('_c, _res', [
        ("####", [0.0, 0.0, 0.0, 0.0]),
        ("##1#", [0.0, 0.0, None, 0.0]),
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

    @pytest.mark.parametrize('_init_eis, _res', [
        (0.0, 0.1),
        (0.1, 0.19),
        (0.5, 0.55),
        (1.0, 1.0),
    ])
    def test_should_increase_eis(self, _init_eis, _res, idx=0, beta=0.1):
        # given
        cond = Condition('####')
        cond[idx].eis = _init_eis

        # when
        cond.increase_eis(idx, beta)

        # then
        assert cond[idx].eis == _res

    @pytest.mark.parametrize('_init_eis, _res', [
        (0.7, 0.63),
        (0.3, 0.27),
        (0, 0),
    ])
    def test_should_decrease_eis(self, _init_eis, _res, idx=0, beta=0.1):
        # given
        cond = Condition('####')
        cond[idx].eis = _init_eis

        # when
        cond.decrease_eis(idx, beta)

        # then
        assert cond[idx].eis == _res
