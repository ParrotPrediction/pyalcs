import pytest

from lcs import Perception
from lcs.agents.xcs.Condition import Condition


class TestCondition:

    def test_should_get_initialized_with_str(self):
        # given
        condition = Condition("#1O##O##")

        # then
        assert len(condition) == 8

    def test_equal(self):
        assert Condition('########') == Condition('########')
        assert Condition('1#######') != Condition('########')
        assert Condition('########') != Condition('#######1')
        assert Condition('1111####') == Condition('1111####')
        assert Condition('1111####') != Condition('1011####')
        assert Condition('1101####') != Condition('1111####')
        assert Condition('00001###') == Condition('00001###')

        assert Condition("11##") == Perception("11##")
        # str doesn't have items. I assume it is purposeful.

    def test_should_hash(self):
        assert hash(Condition('111')) == hash(Condition('111'))
        assert hash(Condition('111')) != hash(Condition('112'))

    def test_subsumes(self):
        assert Condition("1111").subsumes(Perception("1111"))
        assert Condition("11##").subsumes(Perception("1111"))
        assert Condition("11##").subsumes(Perception("11##"))
        assert Condition("11").subsumes(Perception("1111"))
        assert not Condition("1111").subsumes(Perception("10##"))

        assert Condition("11##").subsumes(Condition("11##"))
        assert Condition("11##").subsumes(Perception("11##"))
        assert Condition("11##").subsumes("11##")

    def test_number_of_wildcards(self):
        assert Condition("1011").wildcard_number() == 0
        assert Condition("101#").wildcard_number() == 1
        assert Condition("10##").wildcard_number() == 2
        assert Condition("1###").wildcard_number() == 3
        assert Condition("1##0").wildcard_number() == 2
        assert Condition("1#10").wildcard_number() == 1

    def test_is_more_general(self):
        assert Condition("1111").is_more_general(Condition("1111"))
        assert Condition("11##").is_more_general(Condition("1111"))
        assert Condition("####").is_more_general(Condition("##11"))
        assert not Condition("1100").is_more_general(Condition("1111"))
        assert not Condition("1100").is_more_general(Condition("11##"))
