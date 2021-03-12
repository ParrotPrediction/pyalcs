import random

import pytest

from lcs import Perception
from lcs.agents.macs.macs import Condition


class TestCondition:

    def test_should_get_initialized(self):
        # given
        condition = Condition("#1O#")

        # then
        assert len(condition) == 4
        assert all(type(i) is str for i in condition)
        assert condition[0] == Condition.WILDCARD
        assert condition[3] == Condition.WILDCARD
        assert condition.eis == [0.5, 0.5, 0.5, 0.5]
        assert condition.ig == [0.5, 0.5, 0.5, 0.5]

    @pytest.mark.parametrize('_c, _res', [
        ("####", [0.5, 0.5, 0.5, 0.5]),
        ("##1#", [0.5, 0.5, 0.5, 0.5]),
    ])
    def test_should_get_expected_improvements_property(self, _c, _res):
        assert Condition(_c).eis == _res

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
        cond.eis[0] = 0.2
        p0 = Perception('0000')

        assert cond.does_match(p0)

    @pytest.mark.parametrize('_c, _eis, _res', [
        ('####', [0.2, 0.3, 0.5, 0.4], 2),
        ('##1#', [0.2, 0.3, 0.5, 0.4], 3),
        ('1###', [0.5, 0.5, 0.5, 0.5], 1),
        ('1111', [0.2, 0.3, 0.5, 0.4], None),
    ])
    def test_should_return_index_to_specialize(self, _c, _eis, _res):
        cond = Condition(_c)
        cond.eis = _eis

        assert cond.feature_to_specialize() == _res

    @pytest.mark.parametrize('_c, _ig, _res', [
        ('####', [0.2, 0.3, 0.5, 0.4], None),
        ('##1#', [0.2, 0.3, 0.5, 0.4], 2),
        ('1#1#', [0.5, 0.5, 0.6, 0.5], 2),
        ('1111', [0.2, 0.3, 0.5, 0.4], 2),
    ])
    def test_should_return_index_to_generalized(self, _c, _ig, _res):
        cond = Condition(_c)
        cond.ig = _ig

        assert cond.feature_to_generalize() == _res

    def test_should_generate_generalized_conditions(self):
        # given
        cond = Condition('##12')

        # when
        new_conds = [c for c, idx in cond.exhaustive_generalization()]

        # then
        assert len(new_conds) == 2
        assert all(c != cond for c in new_conds)
        assert all(c is not cond for c in new_conds)
        assert cond == Condition('##12')
        assert new_conds[0] == Condition('###2')
        assert new_conds[1] == Condition('##1#')

    @pytest.mark.parametrize("_cond, _res", [
        ('####', 0),
        ('###1', 1),
        ('2121', 4),
    ])
    def test_should_generate_proper_number_of_new_conditions(self, _cond, _res):
        cond = Condition(_cond)
        assert len([c for c in cond.exhaustive_generalization()]) == _res

    @pytest.mark.parametrize('_self, _other, _res', [
        ('####', '####', True),
        ('####', '###1', False),
        ('###1', '###1', True),
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
        for idx, t in enumerate(cond):
            if t == Condition.WILDCARD and random.random() < 0.5:
                cond.eis[idx] = random.random()

        for idx, t in enumerate(other):
            if t == Condition.WILDCARD and random.random() < 0.5:
                other.eis[idx] = random.random()

        # then
        assert cond.is_more_general(other) == _res

    @pytest.mark.parametrize('_init, _idx, _param, _res', [
        (0.0, 0, 'eis', 0.1), (0.0, 1, 'ig', 0.1),
        (0.1, 0, 'eis', 0.19), (0.1, 1, 'ig', 0.19),
        (0.5, 0, 'eis', 0.55), (0.5, 1, 'ig', 0.55),
        (1.0, 0, 'eis', 1.0), (1.0, 1, 'ig', 1.0),
    ])
    def test_should_increase_by_widrow_hoff(self, _init, _res, _idx,
                                            _param, beta=0.1):
        # given
        cond = Condition('#2#')

        assert _param in ['eis', 'ig']

        # when & then
        if _param == 'eis':
            cond.eis[_idx] = _init
            cond.increase_eis(_idx, beta)
            assert cond.eis[_idx] == _res
        elif _param == 'ig':
            cond.ig[_idx] = _init
            cond.increase_ig(_idx, beta)
            assert cond.ig[_idx] == _res

    @pytest.mark.parametrize('_init, _idx, _param, _res', [
        (0.7, 0, 'eis', 0.63), (0.7, 1, 'ig', 0.63),
        (0.3, 0, 'eis', 0.27), (0.3, 1, 'ig', 0.27),
        (0, 0, 'eis', 0), (0, 1, 'ig', 0)
    ])
    def test_should_decrease_by_widrow_hoff(self, _init, _res, _idx,
                                            _param, beta=0.1):
        # given
        cond = Condition('#2##')

        assert _param in ['eis', 'ig']

        if _param == 'eis':
            cond.eis[_idx] = _init
            cond.decrease_eis(_idx, beta)
            assert cond.eis[_idx] == _res
        elif _param == 'ig':
            cond.ig[_idx] = _init
            cond.decrease_ig(_idx, beta)
            assert cond.ig[_idx] == _res
