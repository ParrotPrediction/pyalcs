import pytest

from lcs.representations import UBR


class TestUBR:

    def test_should_compare_without_ordering(self):
        # given
        o1 = UBR(0, 2)
        o2 = UBR(2, 0)

        # then
        assert o1 == o2
        assert (o1 != o2) is False

    @pytest.mark.parametrize("items, expected_length", [
        ([UBR(2, 5), UBR(5, 2)], 1),
        ([UBR(2, 5), UBR(2, 5)], 1),
        ([UBR(2, 5), UBR(2, 6)], 2)
    ])
    def test_should_not_allow_duplicates(self, items, expected_length):
        # given
        container = set(items)

        # then
        assert len(container) == expected_length

    @pytest.mark.parametrize("_ubr, _span", [
        (UBR(0, 15), 16),
        (UBR(5, 5), 1),
        (UBR(5, 6), 2),
    ])
    def test_should_calculate_bound_span(self, _ubr, _span):
        assert _ubr.bound_span == _span

    @pytest.mark.parametrize("ubr1, ubr2, _result", [
        (UBR(0, 15), UBR(2, 4), True),
        (UBR(4, 6), UBR(6, 7), False),
        (UBR(4, 6), UBR(4, 5), True),
        (UBR(4, 4), UBR(4, 4), True),
        (UBR(4, 4), UBR(4, 5), False),
        (UBR(4, 5), UBR(2, 7), False),
    ])
    def test_should_detect_incorporation(self, ubr1, ubr2, _result):
        assert ubr1.incorporates(ubr2) == _result

    @pytest.mark.parametrize("ubr1, ubr2, _result", [
        (UBR(2, 4), UBR(5, 9), False),
        (UBR(1, 4), UBR(4, 6), True),
        (UBR(5, 6), UBR(2, 5), True),
        (UBR(1, 10), UBR(2, 3), False),
        (UBR(1, 10), UBR(2, 3), False),
        (UBR(5, 2), UBR(7, 5), True),
    ])
    def test_detect_if_can_be_merged(self, ubr1, ubr2, _result):
        assert ubr1.can_be_merged(ubr2) == _result
        assert ubr2.can_be_merged(ubr1) == _result
