import pytest

from lcs.representations import UBR
from lcs.representations.utils import add_from_both_sides


class TestUtils:

    @pytest.mark.parametrize("_ubr, _delta, _result", [
        (UBR(3, 5), 2, UBR(1, 7)),
        (UBR(0, 2), 1, UBR(0, 3)),
        (UBR(0, 15), 5, UBR(0, 15)),
        (UBR(4, 4), 1, UBR(3, 5)),
        (UBR(6, 2), 1, UBR(1, 7)),
        (UBR(6, 2), 0, UBR(6, 2)),
    ])
    def test_should_add_value_to_ubr(self, _ubr, _delta, _result):
        # given
        lb, ub = 0, 15

        # when
        add_from_both_sides(_ubr, _delta, lb, ub)

        # then
        assert _result == _ubr
