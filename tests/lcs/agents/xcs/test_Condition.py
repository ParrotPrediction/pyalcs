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

    def test_should_hash(self):
        assert hash(Condition('111')) == hash(Condition('111'))
        assert hash(Condition('111')) != hash(Condition('112'))

    def test_subsumes(self):
        assert Condition("1111").subsumes(Perception("1111"))
        assert Condition("11##").subsumes(Perception("1111"))
        assert Condition("11##").subsumes(Perception("11##"))
        assert Condition("11").subsumes(Perception("1111"))
        assert not Condition("1111").subsumes(Perception("10##"))
