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

    @pytest.mark.parametrize('_p1, _c, _res', [
        ('1010', 1, ['1???', '?0??', '??1?', '???0']),
        ('1010', 2, ['10??', '1?1?', '1??0', '?01?', '??10', '?0?0']),
        ('1010', 3, ['101?', '10?0', '1?10', '?010']),
        ('1010', 4, ['1010']),
    ])
    def test_should_generate(self, _p1, _c, _res):
        # given
        p = Perception(_p1)

        # when
        effects = Effect.generate(p, int(_c))
        effects = list(map(str, effects))

        assert len(effects) == len(_res)
        assert sorted(effects) == sorted(_res)

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
