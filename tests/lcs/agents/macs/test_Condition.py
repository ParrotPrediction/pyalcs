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
