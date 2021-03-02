import pytest

from lcs import Perception
from lcs.agents.macs.macs import Effect


class TestEffect:

    def test_should_initialize(self):
        effect = Effect('1???')
        assert len(effect) == 4

    def test_should_build_empty(self):
        assert Effect.empty(4) == Effect('????')

    @pytest.mark.parametrize('_e, _p, _res', [
        ('????', '1111', True),
        ('??1?', '1111', True),
        ('??1?', '1211', True),
        ('??12', '1212', True),
        ('??2?', '1111', False),
    ])
    def test_should_match(self, _e, _p, _res):
        assert Effect(_e).does_match(Perception(_p)) == _res
