import pytest

from lcs.agents.xncs import Effect


class TestEffect:

    def test_init(self):
        ef = Effect("########")
        assert len(ef) == 8
        assert all(item == "#" for item in ef)

    @pytest.mark.parametrize("cond1, cond2, result", [
        ("1111", "1111", True),
        ("11##", "1111", True),
        ("11##", "11##", True),
        ("11", "1111", True),
        ("1111", "10##", False),
    ])
    def test_subsumes(self, cond1, cond2, result):
        assert result == Effect(cond1).subsumes(cond2)
        assert result == Effect(cond1).subsumes(cond2)
        assert result == Effect(cond1).subsumes(cond2)
        assert result == Effect(cond1).subsumes(cond2)

    @pytest.mark.parametrize("cond1, cond2, result", [
        ("1111", "1111", True),
        ("11##", "1111", False),
        ("1100", "1111", False),
        ("1100", "0111", False),
        ("1100", "0110", False),
        ("11##", "11##", True),
        ("11", "1111", False),
        ("1111", "10##", False),
    ])
    def test_equals(self, cond1, cond2, result):
        assert result == (Effect(cond1) == Effect(cond2))
