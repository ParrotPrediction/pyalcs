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
        condition.specialize(2, '1')
        condition.specialize(5, '0')

        # then
        assert 2 == condition.specificity

    def test_should_get_initialized_with_str(self):
        # given
        condition = Condition("#1O##O##")

        # then
        assert 8 == len(condition)

    def test_should_specialize_1(self):
        # given
        cond = Condition.empty(8)
        diff = Condition('#0###1#1')

        # when
        cond.specialize(new_condition=diff)

        # then
        assert Condition('#0###1#1') == cond

    def test_should_specialize_2(self):
        # given
        c = Condition('###10#1#')
        diff = Condition('010##1##')

        # when
        c.specialize(new_condition=diff)

        # then
        assert Condition('0101011#') == c

    def test_should_specialize_3(self):
        # given
        c = Condition('#101#10#')
        diff = Condition('####1##1')

        # when
        c.specialize(new_condition=diff)

        # then
        assert Condition('#1011101') == c

    def test_should_specialize_4(self):
        # given
        c = Condition('####01#1')
        diff = Condition('2#00####')

        # when
        c.specialize(new_condition=diff)

        # then
        assert Condition(['2', '#', '0', '0', '0', '1', '#', '1']) == c

    def test_should_specialize_5(self):
        # given
        c = Condition(['#', '#', '#', '0', '1', '#', '0', '#'])
        diff = Condition(['1', '0', '1', '#', '#', '0', '#', '#'])

        # when
        c.specialize(new_condition=diff)

        # then
        assert Condition(['1', '0', '1', '0', '1', '0', '0', '#']) == c

    def test_should_match_perception(self):
        # given
        c = Condition.empty(8)
        p = Perception(['1', '0', '0', '1', '1', '0', '0', '1'])

        # General condition - should match everything
        assert c.does_match(p) is True

        # Correct first position
        c.specialize(0, '1')
        assert c.does_match(p) is True

        # Expects 0 as the first condition
        c.specialize(0, '0')
        assert c.does_match(p) is False

        # Should fail when perception length is different
        with pytest.raises(ValueError):
            c.does_match(Perception(['1', '2']))

    def test_should_match_condition_1(self):
        c_empty = Condition.empty(8)
        c = Condition(['1', '0', '0', '1', '1', '0', '0', '1'])

        # General condition - should match everything
        assert c_empty.does_match(c) is True

        # Correct first position
        c_empty.specialize(0, '1')
        assert c_empty.does_match(c) is True

        # Expects 0 as the first condition
        c_empty.specialize(0, '0')
        assert c_empty.does_match(c) is False

    def test_should_match_condition_2(self):
        # Given
        c = Condition('####O###')
        other = Condition('#1O##O##')

        # When
        res = c.does_match(other)

        # Then
        assert res is True