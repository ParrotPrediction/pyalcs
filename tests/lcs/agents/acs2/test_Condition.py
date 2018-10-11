import pytest

from lcs import Perception
from lcs.agents.acs2 import Condition


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

    def test_should_generalize(self):
        # given
        s = "#1O##O##"
        cond = Condition(s)

        # when
        cond.generalize(position=1)

        # then
        assert Condition("##O##O##") == cond

    @pytest.mark.parametrize("_condition, _spec_before, _spec_after", [
        ('##11####', 2, 1),
        ('#######2', 1, 0),
        ('########', 0, 0),
    ])
    def test_should_generalize_specific_attributes_randomly(
            self, _condition, _spec_before, _spec_after):

        # given
        condition = Condition(_condition)
        assert condition.specificity == _spec_before

        # when
        condition.generalize_specific_attribute_randomly()

        # then
        assert condition.specificity == _spec_after

    def test_generalize_decrements_specificity(self):
        # given
        condition = Condition('#11#####')
        assert 2 == condition.specificity

        # when
        condition.generalize(1)

        # then
        assert 1 == condition.specificity

    def test_should_only_accept_strings(self):
        condition = Condition([])

        with pytest.raises(TypeError):
            # Try to store an integer
            condition[0] = 1

    def test_should_initialize_two_times_the_same_way(self):
        # given
        c1 = Condition("#1O##O##")
        c2 = Condition(['#', '1', 'O', '#', '#', 'O', '#', '#'])

        # then
        assert c1 == c2

    def test_should_return_number_of_specified_elements(self):
        # given
        condition = Condition.empty(8)
        assert 0 == condition.specificity

        # when
        condition[2] = '1'
        condition[5] = '0'

        # then
        assert condition.specificity == 2

    @pytest.mark.parametrize("_condition, _diff, _result", [
        ('########', '#0###1#1', '#0###1#1'),
        ('###10#1#', '010##1##', '0101011#'),
        ('#101#10#', '####1##1', '#1011101'),
        ('####01#1', '2#00####', '2#0001#1'),
        ('###01#0#', '101##0##', '1010100#'),
    ])
    def test_should_specialize_with_condition(
            self, _condition, _diff, _result):

        # given
        cond = Condition(_condition)
        diff = Condition(_diff)

        # when
        cond.specialize_with_condition(diff)

        # then
        assert cond == Condition(_result)

    @pytest.mark.parametrize("_c, _p, _result", [
        ('########', '10011001', True),
        ('1#######', '10011001', True),
        ('0#######', '10011001', False),
    ])
    def test_should_match_perception(self, _c, _p, _result):
        # given
        c = Condition(_c)
        p = Perception(_p)

        # then
        assert c.does_match(p) is _result

    @pytest.mark.parametrize("_c, _other, _result", [
        ('########', '10011001', True),
        ('1#######', '10011001', True),
        ('0#######', '10011001', False),
        ('####0###', '#1O##O##', True),
    ])
    def test_should_match_condition(self, _c, _other, _result):
        # given
        c = Condition(_c)
        other = Condition(_other)

        # then
        assert c.does_match(other) is _result

    def test_get_backwards_anticipation(self):
        # given
        p0 = Perception(['1', '1', '1', '1', '1', '0', '1', '1'])
        condition = Condition(['#', '#', '0', '#', '#', '1', '#', '#'])

        # when
        result = condition.get_backwards_anticipation(p0)

        # then
        assert result == ['1', '1', '0', '1', '1', '1', '1', '1']

    def test_get_backwards_anticipation_2(self):
        # given
        p0 = Perception(['0', '1', '1', '1', '1', '0', '1', '0'])
        condition = Condition(['#', '0', '#', '#', '#', '1', '#', '#'])

        # when
        result = condition.get_backwards_anticipation(p0)

        # then
        assert result == ['0', '0', '1', '1', '1', '1', '1', '0']
