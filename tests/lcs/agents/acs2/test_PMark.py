import pytest

from lcs import Perception
from lcs.agents.acs2 import Configuration, PMark, Condition


class TestPMark:

    @pytest.fixture
    def cfg(self):
        return Configuration(8, 8)

    def test_should_initialize_mark(self, cfg):
        mark = PMark(cfg)

        assert 8 == len(mark)
        for m in mark:
            assert 0 == len(m)

    def test_should_detect_if_marked(self, cfg):
        mark = PMark(cfg)
        assert mark.is_marked() is False

        # Add some mark
        mark[1].add('0')
        assert mark.is_marked() is True

    def test_should_set_single_mark(self, cfg):
        mark = PMark(cfg)
        mark[1].add('0')

        assert len(mark) == 8
        assert len(mark[1]) == 1
        assert '0' in mark[1]

        # Try to add the mark one more time into the same position
        mark[1].add('1')
        assert len(mark[1]) == 2
        assert '0' in mark[1]
        assert '1' in mark[1]

        # Check if duplicates are avoided
        mark[1].add('1')
        assert len(mark[1]) == 2
        assert '0' in mark[1]
        assert '1' in mark[1]

    def test_should_complement_mark_from_perception(self, cfg):
        # Given
        p0 = Perception(['0', '1', '1', '1', '0', '1', '1', '1'])
        mark = PMark(cfg)
        mark[0].add('1')
        mark[2].add('1')
        mark[3].add('1')
        mark[6].add('1')

        # When
        mark.complement_marks(p0)

        # Then
        assert 2 == len(mark[0])
        assert '0' in mark[0]
        assert '1' in mark[0]

        assert 1 == len(mark[2])
        assert '1' in mark[2]

        assert 1 == len(mark[3])
        assert '1' in mark[3]

        assert 1 == len(mark[6])
        assert '1' in mark[6]

    @pytest.mark.parametrize("_p0", [
        (['0', '1', '1', '0', '0', '0', '1', '1']),
        (['1', '0', '1', '0', '1', '0', '0', '1']),
    ])
    def test_should_get_differences_1(self, _p0, cfg):
        # given
        generic_condition = Condition.empty(
            wildcard=cfg.classifier_wildcard,
            length=cfg.classifier_length)
        p0 = Perception(_p0)
        mark = PMark(cfg)

        # when
        diff = mark.get_differences(p0)

        # then
        assert diff == generic_condition

    def test_should_get_differences_2(self, cfg):
        # Given
        p0 = Perception(['1', '1', '0', '1', '1', '1', '0', '1'])
        mark = PMark(cfg)
        mark[0].add('1')
        mark[1].add('1')
        mark[2].add('0')
        mark[3].add('0')
        mark[4].add('0')
        mark[5].add('0')
        mark[6].add('1')
        mark[7].add('0')

        for _ in range(100):
            # When
            diff = mark.get_differences(p0)

            # Then
            assert diff is not None
            assert '#' == diff[0]
            assert '#' == diff[1]
            assert '#' == diff[2]
            assert 1 == diff.specificity

    def test_should_get_differences_3(self, cfg):
        # Given
        p0 = Perception(['0', '1', '1', '0', '0', '0', '0', '0'])
        mark = PMark(cfg)
        mark[0].update(['0', '1'])
        mark[1].update(['1'])
        mark[2].update(['0', '1'])
        mark[3].update(['1'])
        mark[4].update(['0', '1'])
        mark[5].update(['1'])
        mark[6].update(['0', '1'])
        mark[7].update(['1'])

        for _ in range(100):
            # When
            diff = mark.get_differences(p0)

            # Then
            assert diff is not None
            assert '#' == diff[0]
            assert '#' == diff[1]
            assert '#' == diff[2]
            assert '#' == diff[4]
            assert '#' == diff[6]
            assert 1 == diff.specificity

    def test_should_get_differences_4(self, cfg):
        # given
        p0 = Perception(['1', '1', '1', '1', '1', '0', '1', '0'])
        mark = PMark(cfg)
        mark[0].update(['0', '1'])
        mark[1].update(['0', '1'])
        mark[3].update(['0', '1'])
        mark[4].update(['0', '1'])
        mark[6].update(['0', '1'])
        mark[7].update(['0'])

        # when
        diff = mark.get_differences(p0)

        # then
        assert diff is not None
        assert 5 == diff.specificity
        assert '1' == diff[0]
        assert '1' == diff[1]
        assert '#' == diff[2]
        assert '1' == diff[3]
        assert '1' == diff[4]
        assert '#' == diff[5]
        assert '1' == diff[6]
        assert '#' == diff[7]

    def test_should_get_differences_5(self, cfg):
        # Given
        p0 = Perception(['0', '0', '2', '1', '1', '0', '1', '0'])
        mark = PMark(cfg)
        mark[3].add('0')
        mark[6].add('0')

        for _ in range(100):
            # When
            diff = mark.get_differences(p0)

            # Then
            assert diff is not None
            assert 1 == diff.specificity
