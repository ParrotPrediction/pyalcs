import pytest

from lcs import Perception
from lcs.agents.yacs.yacs import Effect


class TestEffect:

    def test_should_get_initialized_with_str(self):
        # given
        effect = Effect("#1O##O##")

        # then
        assert len(effect) == 8

    @pytest.mark.parametrize('_p0, _p1, _res', [
        ('1100', '1100', '####'),
        ('1101', '1100', '###0'),
        ('1100', '0011', '0011')
    ])
    def test_should_calculate_desired_effect(self, _p0, _p1, _res):
        assert Effect.diff(Perception(_p0), Perception(_p1)) == Effect(_res)
