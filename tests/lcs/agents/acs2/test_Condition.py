import pytest

from lcs import Perception
from lcs.agents.acs2 import Condition


class TestCondition:

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

    def test_should_get_initialized_with_str(self):
        # given
        condition = Condition("#1O##O##")

        # then
        assert len(condition) == 8

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

    def test_should_match_perception(self):
        # given
        c = Condition.empty(8)
        p = Perception(['1', '0', '0', '1', '1', '0', '0', '1'])

        # General condition - should match everything
        assert c.does_match(p) is True

        # Correct first position
        c[0] = '1'
        assert c.does_match(p) is True

        # Expects 0 as the first condition
        c[0] = '0'
        assert c.does_match(p) is False

    def test_should_match_condition_1(self):
        c_empty = Condition.empty(8)
        c = Condition(['1', '0', '0', '1', '1', '0', '0', '1'])

        # General condition - should match everything
        assert c_empty.does_match(c) is True

        # Correct first position
        c_empty[0] = '1'
        assert c_empty.does_match(c) is True

        # Expects 0 as the first condition
        c_empty[0] = '0'
        assert c_empty.does_match(c) is False

    def test_should_match_condition_2(self):
        # Given
        c = Condition('####O###')
        other = Condition('#1O##O##')

        # When
        res = c.does_match(other)

        # Then
        assert res is True
