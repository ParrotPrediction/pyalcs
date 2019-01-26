import pytest

from lcs.representations import Interval


class TestInterval:

    def test_should_compare_without_ordering(self):
        # given
        i1 = Interval(0.0, 0.2)
        i2 = Interval(0.2, 0.0)

        # then
        assert i1 == i2
        assert (i1 != i2) is False

    @pytest.mark.parametrize("items, expected_length", [
        ([Interval(.2, .5), Interval(.5, .2)], 1),
        ([Interval(.2, .5), Interval(.2, .5)], 1),
        ([Interval(.2, .5), Interval(.2, .6)], 2)
    ])
    def test_should_not_allow_duplicates(self, items, expected_length):
        # given
        container = set(items)

        # then
        assert len(container) == expected_length

    @pytest.mark.parametrize("x1, x2", [
        (1.1, 0.2),
        (0.2, 1.1),
        (-0.1, 0.5),
    ])
    def test_should_not_initialize_due_to_wrong_range(self, x1, x2):
        with pytest.raises(ValueError,
                           match='Value [-]?.{4} not in range \[0, 1\]'):
            Interval(x1, x2)

    @pytest.mark.parametrize("x1, x2, left", [
        (0.2, 0.5, 0.2),
        (0.5, 0.2, 0.2),
    ])
    def test_should_determine_left_bound(self, x1, x2, left):
        assert Interval(x1, x2).left_bound == left

    @pytest.mark.parametrize("x1, x2, right", [
        (0.2, 0.5, 0.5),
        (0.5, 0.2, 0.5),
    ])
    def test_should_determine_right_bound(self, x1, x2, right):
        assert Interval(x1, x2).right_bound == right

    @pytest.mark.parametrize("_i, _span", [
        (Interval(.0, 1.), 1),
        (Interval(.5, .5), 0),
        (Interval(.5, .6), .1),
    ])
    def test_should_calculate_bound_span(self, _i, _span):
        assert abs(_i.span - _span) < 0.00001

    @pytest.mark.parametrize("i1, i2, _result", [
        (Interval(.0, 1.), Interval(.2, .4), True),
        (Interval(.4, .6), Interval(.6, .7), False),
        (Interval(.4, .6), Interval(.4, .5), True),
        (Interval(.4, .4), Interval(.4, .4), True),
        (Interval(.4, .4), Interval(.4, .5), False),
        (Interval(.4, .5), Interval(.2, .7), False),
        (Interval(.4, .6), Interval(.5, .7), False),
    ])
    def test_should_detect_incorporation(self, i1, i2, _result):
        assert (i2 in i1) == _result

    @pytest.mark.parametrize("_interval, _val", [
        (Interval(0.2, 0.5), 0.3),
        (Interval(0.53, 0.52), 0.525)
    ])
    def test_should_detect_value_in_interval(self, _interval, _val):
        assert _val in _interval

    @pytest.mark.parametrize("_i1, _i2, _the_same", [
        (Interval(.1, .2), Interval(.1, .2), True),
        (Interval(.1, .2), Interval(.2, .3), False),
        (Interval(.1, .2), Interval(.101, .201), True),
    ])
    def test_should_compare_intervals(self, _i1, _i2, _the_same):
        assert (_i1 == _i2) == _the_same
