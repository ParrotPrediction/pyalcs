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

    @pytest.mark.parametrize('_e1, _e2, _res', [
        ('????', '????', False),
        ('??1?', '??1?', False),
        ('??1?', '??2?', True),
        ('2?1?', '2?1?', False),
        ('2?1?', '2?2?', True),
        ('2???', '??2?', False),
    ])
    def test_should_detect_conflicts(self, _e1, _e2, _res):
        assert Effect(_e1).conflicts(Effect(_e2)) == _res
        assert Effect(_e2).conflicts(Effect(_e1)) == _res

    def test_should_sort(self):
        # given
        e1 = Effect('???1')
        e2 = Effect('?2?1')
        e3 = Effect('???1')

        effects = [e1, e2, e3]
        assert effects == [e1, e2, e3]

        # when
        effects.sort()

        # then
        assert effects == [e2, e1, e3]
