import pytest

from lcs import Perception
from lcs.agents.racs import Configuration, Mark, Condition
from lcs.representations import UBR


class TestMark:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder_bits=4)

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
        mark[0].add(UBR(2, 5))

        # then
        assert mark.is_marked() is True

    @pytest.mark.parametrize("initmark, perception, changed", [
        ([[], []], [0.5, 0.5], False),  # shouldn't set mark if empty
        ([[8], []], [0.5, 0.5], False),  # encoded value already marked
        ([[5], []], [0.5, 0.5], True)
    ])
    def test_should_complement_mark(self, initmark, perception, changed, cfg):
        # given
        p0 = Perception(perception, oktypes=(float,))
        mark = self._init_mark(initmark, cfg)

        # when
        change_detected = mark.complement_marks(p0)

        # then
        assert change_detected is changed

    @pytest.mark.parametrize("initmark, perc, initcond, marked_count", [
        # not marked, all generic classifier, should mark two positions
        ([[], []], [0.5, 0.5], [], 2),
        # not marked, specified condition, shouldn't get marked
        ([[], []], [0.5, 0.5], [UBR(1, 3), UBR(2, 3)], 0),
        # not marked, one don't care, should mark one
        ([[], []], [0.5, 0.5], [UBR(1, 3), UBR(0, 16)], 1),
        # already marked, should use perception, one mark
        ([[4], []], [0.5, 0.5], [], 1),
    ])
    def test_should_set_mark_using_condition(self, initmark, perc,
                                             initcond, marked_count, cfg):
        # given
        p0 = Perception(perc, oktypes=(float,))
        mark = self._init_mark(initmark, cfg)
        condition = self._init_condition(initcond, cfg)

        # when
        mark.set_mark_using_condition(condition, p0)

        # then
        assert self._count_marked_attributes(mark) is marked_count

    @staticmethod
    def _init_mark(vals, cfg):
        mark = Mark(cfg)
        for idx, attribs in enumerate(vals):
            for attrib in attribs:
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
