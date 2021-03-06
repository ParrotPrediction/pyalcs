import pytest

from lcs import Perception
from lcs.agents.macs.macs import Condition, DontCare


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
        assert condition[0].eis == 0.5
        assert condition[3].eis == 0.5

    @pytest.mark.parametrize('_c, _res', [
        ("####", [0.5, 0.5, 0.5, 0.5]),
        ("##1#", [0.5, 0.5, 0.5, 0.5]),
    ])
    def test_should_get_expected_improvements_property(self, _c, _res):
        assert Condition(_c).expected_improvements == _res

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

    @pytest.mark.parametrize('_c, _eis, _res', [
        ('####', [0.2, 0.3, 0.5, 0.4], 2),
        ('##1#', [0.2, 0.3, 0.5, 0.4], 3),
        ('1###', [0.5, 0.5, 0.5, 0.5], 1),
        ('1111', [0.2, 0.3, 0.5, 0.4], None),
    ])
    def test_should_return_index_to_specialize(self, _c, _eis, _res):
        cond = Condition(_c)
        for c, eis in zip(cond, _eis):
            if hasattr(c, 'eis'):
                c.eis = eis

        assert cond.feature_to_specialize() == _res

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
