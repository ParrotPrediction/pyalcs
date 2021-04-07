import pytest

from lcs.agents.xncs import Effect


class TestEffect:

    def test_init(self):
        ef = Effect("########")
        assert len(ef) == 8
        assert all(item == "#" for item in ef)
