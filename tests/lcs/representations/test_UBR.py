import pytest

from lcs.representations import UBR

DELTA = 0.00001


class TestUBR:

    def test_should_compare_without_ordering(self):
        # given
        o1 = UBR(0.0, 0.2)
        o2 = UBR(0.2, 0.0)

        # then
        assert o1 == o2
        assert (o1 != o2) is False

    @pytest.mark.parametrize("items, expected_length", [
        ([UBR(.2, .5), UBR(.5, .2)], 1),
        ([UBR(.2, .5), UBR(.2, .5)], 1),
        ([UBR(.2, .5), UBR(.2, .6)], 2)
    ])
    def test_should_not_allow_duplicates(self, items, expected_length):
        # given
        container = set(items)

        # then
        assert len(container) == expected_length

    @pytest.mark.parametrize("_ubr, _span", [
        (UBR(0, 1), 1),
        (UBR(.5, .5), 0),
        (UBR(.5, .6), .1),
    ])
    def test_should_calculate_bound_span(self, _ubr, _span):
        assert abs(_ubr.bound_span - _span) < DELTA

    @pytest.mark.parametrize("ubr1, ubr2, _result", [
        (UBR(0, 1), UBR(.2, .4), True),
        (UBR(.4, .6), UBR(.6, .7), False),
        (UBR(.4, .6), UBR(.4, .5), True),
        (UBR(.4, .4), UBR(.4, .4), True),
        (UBR(.4, .4), UBR(.4, .5), False),
        (UBR(.4, .5), UBR(.2, .7), False),
        (UBR(.4, .6), UBR(.5, .7), False),
    ])
    def test_should_detect_incorporation(self, ubr1, ubr2, _result):
        assert ubr1.incorporates(ubr2) == _result
