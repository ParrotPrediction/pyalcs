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

    @pytest.mark.parametrize('_c, _eis, _res', [
        ("####", [.4, .1, .2, 0], 0),
        ("1###", [None, .1, .2, .05], 2),
        ("112#", [None, None, None, .05], 3),
        ("1122", [None, None, None, None], None),
    ])
    def test_should_return_token_idx_to_specialize(self, _c, _eis, _res):
        # given
        c = Condition(_c)
        for idx, ei in enumerate(_eis):
            if ei is not None:
                c[idx].eis = ei

        # then
        assert c.token_idx_to_specialize() == _res
