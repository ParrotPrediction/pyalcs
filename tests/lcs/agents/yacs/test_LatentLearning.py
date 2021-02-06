import random

import pytest

from lcs import Perception
from lcs.agents.yacs.yacs import Configuration, Condition, Effect, Classifier, ClassifierTrace, ClassifiersList, LatentLearning


class TestLatentLearning:

    @pytest.fixture
    def cfg(self):
        return Configuration(4, 2,
                             trace_length=5,
                             feature_possible_values=[2, 2, 2, 2])

    @pytest.fixture
    def ll(self, cfg):
        return LatentLearning(cfg)

    def test_effect_covering_should_add_traces(self, cfg, ll):
        # given
        cl1 = Classifier(condition='1###', action=0, effect='##1#', cfg=cfg)
        cl2 = Classifier(condition='1###', action=1, effect='#1#0', cfg=cfg)
        cl3 = Classifier(condition='1###', action=0, effect='####', cfg=cfg)
        cl4 = Classifier(condition='0###', action=0, effect='##1#', cfg=cfg)
        population = ClassifiersList(*[cl1, cl2, cl3, cl4])
        p0 = Perception('1000')
        p1 = Perception('1010')
        prev_action = 0

        # when
        ll.effect_covering(population, p0, p1, prev_action)

        # then
        assert len(population) == 4
        assert list(cl1.trace) == [ClassifierTrace.GOOD]
        assert list(cl2.trace) == []
        assert list(cl3.trace) == [ClassifierTrace.BAD]
        assert list(cl4.trace) == []

    def test_effect_covering_should_add_new_classifier(self, cfg, ll):
        # given
        cl1 = Classifier(condition='1##0', action=0, effect='#1#1', cfg=cfg)
        cl2 = Classifier(condition='1###', action=1, effect='#1#0', cfg=cfg)
        cl3 = Classifier(condition='1###', action=0, effect='####', cfg=cfg)
        cl4 = Classifier(condition='0###', action=0, effect='##1#', cfg=cfg)
        population = ClassifiersList(*[cl1, cl2, cl3, cl4])
        p0 = Perception('1000')
        p1 = Perception('1010')
        prev_action = 0
        random.seed(31337)  # selects c1 as an old classifier

        # when
        ll.effect_covering(population, p0, p1, prev_action)

        # then
        assert len(population) == 5
        assert list(cl1.trace) == [ClassifierTrace.BAD]
        assert list(cl2.trace) == []
        assert list(cl3.trace) == [ClassifierTrace.BAD]
        assert list(cl4.trace) == []

        new_cl = population[4]
        assert new_cl.condition == Condition('1##0')
        assert new_cl.action == 0
        assert new_cl.effect == Effect('##1#')
        assert list(new_cl.trace) == [ClassifierTrace.GOOD]

    def test_select_accurate_classifiers(self, cfg, ll):
        # given
        cl1 = Classifier(cfg=cfg)  # should be kept (trace not full)
        cl1.add_to_trace(ClassifierTrace.GOOD)
        cl1.add_to_trace(ClassifierTrace.BAD)

        cl2 = Classifier(cfg=cfg)  # should be kept (all good traces)
        cl2.add_to_trace(ClassifierTrace.GOOD)
        cl2.add_to_trace(ClassifierTrace.GOOD)
        cl2.add_to_trace(ClassifierTrace.GOOD)
        cl2.add_to_trace(ClassifierTrace.GOOD)
        cl2.add_to_trace(ClassifierTrace.GOOD)

        cl3 = Classifier(cfg=cfg)  # should be removed (trace full, oscillate)
        cl3.add_to_trace(ClassifierTrace.GOOD)
        cl3.add_to_trace(ClassifierTrace.GOOD)
        cl3.add_to_trace(ClassifierTrace.BAD)
        cl3.add_to_trace(ClassifierTrace.GOOD)
        cl3.add_to_trace(ClassifierTrace.GOOD)

        population = ClassifiersList(*[cl1, cl2, cl3])

        # when
        ll.select_accurate_classifiers(population)

        # then
        assert len(population) == 2
        assert cl1 in population
        assert cl2 in population

    def test_mutspec(self, cfg, ll):
        # given
        cl = Classifier(condition='####', action=0, effect='####', cfg=cfg)
        feature_idx = 0

        # when
        new_cls = list(ll.mutspec(cl, feature_idx))

        # then
        assert len(new_cls) == 2
        assert all(True for cl in new_cls if cl.action == 0)
        assert all(True for cl in new_cls if cl.effect == Effect('####'))
        assert new_cls[0].condition == Condition('0###')
        assert new_cls[1].condition == Condition('1###')

    def test_mutspec_with_effect_change(self, cfg, ll):
        # given
        cl = Classifier(condition='####', action=0, effect='1###', cfg=cfg)
        feature_idx = 0

        # when
        new_cls = list(ll.mutspec(cl, feature_idx))

        # then
        assert len(new_cls) == 2
        assert all(True for cl in new_cls if cl.action == 0)
        assert new_cls[0].condition == Condition('0###')
        assert new_cls[0].effect == Effect('1###')
        assert new_cls[1].condition == Condition('1###')
        assert new_cls[1].effect == Effect('####')

