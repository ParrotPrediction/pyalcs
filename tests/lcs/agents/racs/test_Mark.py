import pytest

from lcs import Perception
from lcs.agents.racs import Configuration, Mark, Condition
from lcs.representations import Interval


class TestMark:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2)

    def test_should_initialize_empty_mark(self, cfg):
        # when
        mark = Mark(cfg)

        # then
        assert len(mark) == 2
        for m in mark:
            assert type(m) is set
            assert len(m) == 0

    def test_should_detect_if_not_marked(self, cfg):
        mark = Mark(cfg)
        assert mark.is_marked() is False

    def test_should_detect_if_marked(self, cfg):
        # given
        mark = Mark(cfg)

        # when
        mark[0].add(Interval(.2, .2))

        # then
        assert mark.is_marked() is True

    @pytest.mark.parametrize("initmark, perception, changed", [
        ([[], []], [.5, .5], False),  # shouldn't set mark if not marked
        ([[.5], []], [.5, .5], False),  # perception value already marked
        ([[.2], []], [.5, .5], True)
    ])
    def test_should_complement_mark(self, initmark, perception, changed, cfg):
        # given
        p0 = Perception(perception, oktypes=(float,))
        mark = self._init_mark(initmark, cfg)

        # when
        change_detected = mark.complement_marks(p0)

        # then
        assert change_detected is changed

    @pytest.mark.parametrize("initmark, _p0, initcond, marked_count", [
        # not marked, all generic classifier, should mark two positions
        ([[], []], [.5, .5], [], 2),
        # not marked, specified condition, shouldn't get marked
        ([[], []], [.5, .5], [Interval(.1, .3), Interval(.2, .3)], 0),
        # not marked, one don't care, should mark one
        ([[], []], [.5, .5], [Interval(.1, .3), Interval(0., 1.)], 1),
        # already marked, should use perception, one mark
        ([[.4], []], [.5, .5], [], 1),
    ])
    def test_should_set_mark_using_condition(self, initmark, _p0,
                                             initcond, marked_count, cfg):
        # given
        p0 = Perception(_p0, oktypes=(float,))
        mark = self._init_mark(initmark, cfg)
        condition = self._init_condition(initcond, cfg)

        # when
        mark.set_mark_using_condition(condition, p0)

        # then
        assert self._count_marked_attributes(mark) is marked_count

    def test_should_get_no_differences(self, cfg):
        # given
        p0 = Perception([.5, .5], oktypes=(float,))
        mark = self._init_mark([], cfg)

        # when
        diff = mark.get_differences(p0)

        # then
        assert diff == Condition.generic(cfg)

    @pytest.mark.parametrize("_m, _p0, _specif", [
        # There is no perception in mark - one attribute should be
        # randomly specified
        ([[.2], [.4]], [.5, .5], 1),
        # One perception is marked - the other should be specified
        ([[.5], [.2]], [.5, .5], 1),
        # Both perceptions are marked - no differences
        ([[.5], [.5]], [.5, .5], 0)
    ])
    def test_should_handle_unique_differences(self, _m, _p0, _specif, cfg):
        # given
        p0 = Perception(_p0, oktypes=(float,))
        mark = self._init_mark(_m, cfg)

        # when
        diff = mark.get_differences(p0)

        # then
        assert diff.specificity == _specif

    @pytest.mark.parametrize("_m, _p0, _specificity", [
        # There are two marks in one attribute - it should be specified.
        ([[.1, .2], [.4]], [.5, .5], 1),
        # Here we have clear unique difference - specify it first
        ([[.1, .2], [.5]], [.5, .5], 1),
        # Two fuzzy attributes (containing perception value) - both
        # should be specified
        ([[.4, .5], [.3, .5]], [.5, .5], 2),
        # Two fuzzy attributes - but one is unique (does not contain
        # perception)
        ([[.4, .5], [.4, .6]], [.5, .5], 1),
    ])
    def test_should_handle_fuzzy_differences(self, _m, _p0, _specificity, cfg):
        # given
        p0 = Perception(_p0, oktypes=(float,))
        mark = self._init_mark(_m, cfg)

        # when
        diff = mark.get_differences(p0)

        # then
        assert diff.specificity == _specificity

    @staticmethod
    def _init_mark(vals, cfg):
        mark = Mark(cfg)
        for idx, attribs in enumerate(vals):
            for attrib in attribs:
                assert type(attrib) is float
                mark[idx].add(attrib)

        return mark

    @staticmethod
    def _init_condition(vals, cfg):
        if len(vals) == 0:
            return Condition.generic(cfg)

        return Condition(vals, cfg)

    @staticmethod
    def _count_marked_attributes(mark) -> int:
        return sum(1 for m in mark if len(m) > 0)
