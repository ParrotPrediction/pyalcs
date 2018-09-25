import pytest

from lcs import Perception
from lcs.agents.acs2 import Configuration, ClassifiersList, Classifier
from lcs.strategies.action_planning.goal_sequence_searcher \
    import GoalSequenceSearcher


class TestGoalSequenceSearcher:
    @pytest.fixture
    def cfg(self):
        return Configuration(8, 8)

    def test_does_contain_state(self):
        # given
        s0 = Perception("11111111")
        s1 = Perception("00000000")

        perceptions0 = [s0]
        perceptions1 = [s1]
        perceptions2 = [s1, s0]

        # when
        result_0 = GoalSequenceSearcher.does_contain_state(perceptions0, s0)
        result_none = GoalSequenceSearcher.does_contain_state(perceptions1, s0)
        result_1 = GoalSequenceSearcher.does_contain_state(perceptions2, s0)

        # then
        assert result_0 == 0
        assert result_1 == 1
        assert result_none is None

    def test_search_goal_sequence_1(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        start = Perception("11111111")
        goal = Perception("11111110")

        empty_list = ClassifiersList(cfg=cfg)

        # when
        result = gs.search_goal_sequence(empty_list, start=start, goal=goal)

        assert result == []

    @pytest.mark.skip(reason="not implemented yet")
    def test_search_goal_sequence(self):
        pass

    def test_form_new_classifiers_1(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        cls_list = [ClassifiersList(cfg=cfg)]
        cl = Classifier(condition="01010101", action=2, effect="0000000",
                        cfg=cfg)
        i = 0

        # when
        new_classifiers = gs._form_new_classifiers(cls_list, i, cl, cfg)

        # then
        assert len(new_classifiers) == 1
        assert cl in new_classifiers

    def test_form_new_classifiers_2(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        cl0 = Classifier(condition="01010101", action=2, effect="0000000",
                         cfg=cfg)
        cl1 = Classifier(condition="11111111", action=0, effect="0000000",
                         cfg=cfg)
        cls_list = [ClassifiersList(cl0, cfg=cfg)]
        i = 1

        # when
        new_classifiers = gs._form_new_classifiers(cls_list, i, cl1, cfg)

        # then
        assert len(new_classifiers) == 2
        assert cl0 in new_classifiers
        assert cl1 in new_classifiers

    def test_form_sequence_forwards_1(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        cl0 = Classifier(condition="01010101", action=2, effect="0000000",
                         cfg=cfg)
        i = 0
        idx = 0

        # when
        seq = gs._form_sequence_forwards(i, idx, cl0)

        # then
        assert len(seq) == 1
        assert cl0.action in seq

    def test_form_sequence_forwards_2(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        cl0 = Classifier(condition="01010101", action=2, effect="0000000",
                         cfg=cfg)
        cl1 = Classifier(condition="11111111", action=0, effect="0000000",
                         cfg=cfg)
        gs.forward_classifiers = [ClassifiersList(cl0, cfg=cfg)]
        i = 1
        idx = 0

        # when
        seq = gs._form_sequence_forwards(i, idx, cl1)

        # then
        assert len(seq) == 2
        assert cl0.action in seq
        assert cl1.action in seq
        assert seq == [cl0.action, cl1.action]

    def test_form_sequence_forwards_3(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        cl0 = Classifier(condition="01010101", action=2, effect="0000000",
                         cfg=cfg)
        cl1 = Classifier(condition="11111111", action=0, effect="0000000",
                         cfg=cfg)
        gs.backward_classifiers = [ClassifiersList(cl0, cfg=cfg)]
        i = 0
        idx = 1

        # when
        seq = gs._form_sequence_forwards(i, idx, cl1)

        # then
        assert len(seq) == 2
        assert cl0.action in seq
        assert cl1.action in seq
        assert seq == [cl1.action, cl0.action]

    def test_form_sequence_forwards_4(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        cl0 = Classifier(condition="01010101", action=2, effect="0000000",
                         cfg=cfg)
        cl1 = Classifier(condition="11111111", action=0, effect="0000000",
                         cfg=cfg)
        cl2 = Classifier(condition="11111111", action=1, effect="0000000",
                         cfg=cfg)
        gs.forward_classifiers = [ClassifiersList(cl0, cfg=cfg)]
        gs.backward_classifiers = [ClassifiersList(cl1, cl0, cfg=cfg)]
        i = 1
        idx = 1

        # when
        seq = gs._form_sequence_forwards(i, idx, cl2)

        # then
        assert len(seq) == 4
        assert cl0.action in seq
        assert cl1.action in seq
        assert cl2.action in seq
        assert seq == [cl0.action, cl2.action, cl1.action, cl0.action]

    def test_form_sequence_forwards_5(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        cl0 = Classifier(condition="01010101", action=2, effect="0000000",
                         cfg=cfg)
        cl1 = Classifier(condition="11111111", action=0, effect="0000000",
                         cfg=cfg)
        cl2 = Classifier(condition="11111111", action=1, effect="0000000",
                         cfg=cfg)
        gs.forward_classifiers = [ClassifiersList(cl0, cfg=cfg),
                                  ClassifiersList(cl0, cl1, cfg=cfg)]
        i = 2
        idx = 0

        # when
        seq = gs._form_sequence_forwards(i, idx, cl2)

        # then
        assert len(seq) == 3
        assert cl0.action in seq
        assert cl1.action in seq
        assert cl2.action in seq
        assert seq == [cl1.action, cl0.action, cl2.action]

    def test_form_sequence_backwards_1(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        cl0 = Classifier(condition="01010101", action=2, effect="0000000",
                         cfg=cfg)
        i = 0
        idx = 0

        # when
        seq = gs._form_sequence_backwards(i, idx, cl0)

        # then
        assert len(seq) == 1
        assert cl0.action in seq

    def test_form_sequence_backwards_2(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        cl0 = Classifier(condition="01010101", action=2, effect="0000000",
                         cfg=cfg)
        cl1 = Classifier(condition="11111111", action=0, effect="0000000",
                         cfg=cfg)
        gs.backward_classifiers = [ClassifiersList(cl0, cfg=cfg)]
        i = 1
        idx = 0

        # when
        seq = gs._form_sequence_backwards(i, idx, cl1)

        # then
        assert len(seq) == 2
        assert cl0.action in seq
        assert cl1.action in seq
        assert seq == [cl1.action, cl0.action]

    def test_form_sequence_backwards_3(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        cl0 = Classifier(condition="01010101", action=2, effect="0000000",
                         cfg=cfg)
        cl1 = Classifier(condition="11111111", action=0, effect="0000000",
                         cfg=cfg)
        gs.forward_classifiers = [ClassifiersList(cl0, cfg=cfg)]
        i = 0
        idx = 1

        # when
        seq = gs._form_sequence_backwards(i, idx, cl1)

        # then
        assert len(seq) == 2
        assert cl0.action in seq
        assert cl1.action in seq
        assert seq == [cl0.action, cl1.action]

    def test_form_sequence_backwards_4(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        cl0 = Classifier(condition="01010101", action=2, effect="0000000",
                         cfg=cfg)
        cl1 = Classifier(condition="11111111", action=0, effect="0000000",
                         cfg=cfg)
        cl2 = Classifier(condition="11111111", action=1, effect="0000000",
                         cfg=cfg)
        gs.forward_classifiers = [ClassifiersList(cl0, cfg=cfg)]
        gs.backward_classifiers = [ClassifiersList(cl1, cl0, cfg=cfg)]
        i = 1
        idx = 1

        # when
        seq = gs._form_sequence_backwards(i, idx, cl2)

        # then
        assert len(seq) == 4
        assert cl0.action in seq
        assert cl1.action in seq
        assert cl2.action in seq
        assert seq == [cl0.action, cl2.action, cl1.action, cl0.action]

    def test_form_sequence_backwards_5(self, cfg):
        # given
        gs = GoalSequenceSearcher()
        cl0 = Classifier(condition="01010101", action=2, effect="0000000",
                         cfg=cfg)
        cl1 = Classifier(condition="11111111", action=0, effect="0000000",
                         cfg=cfg)
        cl2 = Classifier(condition="11111111", action=1, effect="0000000",
                         cfg=cfg)
        gs.backward_classifiers = [ClassifiersList(cl0, cfg=cfg),
                                   ClassifiersList(cl0, cl1, cfg=cfg)]
        i = 2
        idx = 0

        # when
        seq = gs._form_sequence_backwards(i, idx, cl2)

        # then
        assert len(seq) == 3
        assert cl0.action in seq
        assert cl1.action in seq
        assert cl2.action in seq
        assert seq == [cl2.action, cl0.action, cl1.action]
