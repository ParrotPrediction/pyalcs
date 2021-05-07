import pytest

from lcs import Perception
from lcs.agents.xcs.Condition import Condition


class TestCondition:

    def test_should_get_initialized_with_str(self):
        # given
        condition = Condition("#1O##O##")

        # then
        assert len(condition) == 8

    @pytest.mark.parametrize("cond1, cond2, result", [
        ("########", "########", True),
        ("1100####", "########", False),
        ("########", "####1100", False),
        ("1111####", "####1111", False),
        ("1111####", "########", False),

        ("1100", "1100", True),
        ("1100", "1111", False),
        ("1111", "1100", False),

        ("1111", "11", False),
        ("#1111#", "#1111#", True),
        ("###01#", "###01#", True),
    ])
    def test_equal(self, cond1, cond2, result):
        assert result == (Condition(cond1) == Condition(cond2))
        assert result == (Condition(cond1) == Perception(cond2))

    def test_should_hash(self):
        assert hash(Condition('111')) == hash(Condition('111'))
        assert hash(Condition('111')) != hash(Condition('112'))

    @pytest.mark.parametrize("cond1, cond2, result", [
        ("1111", "1111", True),
        ("11##", "1111", True),
        ("11##", "11##", True),
        ("11", "1111", True),
        ("1111", "10##", False),
    ])
    def test_subsumes(self, cond1, cond2, result):
        assert result == Condition(cond1).subsumes(Perception(cond2))
        assert result == Condition(cond1).subsumes(Condition(cond2))
        assert result == Condition(cond1).subsumes(Perception(cond2))
        assert result == Condition(cond1).subsumes(cond2)

    @pytest.mark.parametrize("cond, num", [
        ("1011", 0),
        ("101#", 1),
        ("1##1", 2),
        ("##1#", 3),
    ])
    def test_number_of_wildcards(self, cond, num):
        assert Condition(cond).wildcard_number == num

    @pytest.mark.parametrize("cond1, cond2, result", [
        ("1111", "1111", True),
        ("11##", "1111", True),
        ("####", "11##", True),
        ("1100", "####", False),
        ("1111", "10##", False),
    ])
    def test_is_more_general(self, cond1, cond2, result):
        assert Condition(cond1).is_more_general(Condition(cond2)) == result

    @pytest.mark.parametrize("_c", [
        ([0.1, 0.2]),
        ([0, 1]),
    ])
    def test_should_fail_with_invalid_types(self, _c):
        with pytest.raises(AssertionError) as _:
            Condition(_c)
