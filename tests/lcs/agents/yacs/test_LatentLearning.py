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

    def test_should_specialize_conditions(self, cfg, ll):
        # given oscillating classifiers with full traces
        cl1 = Classifier(condition="#1#1", action=0, effect="1122", cfg=cfg)
        cl2 = Classifier(condition="#1#1", action=0, effect="2211", cfg=cfg)
        cl1.condition[0].eis = 0.4
        cl2.condition[0].eis = 0.2

        cl1.condition[2].eis = 0.1
        cl2.condition[2].eis = 0.15

        for i in range(2):
            cl1.add_to_trace(ClassifierTrace.GOOD)
            cl2.add_to_trace(ClassifierTrace.GOOD)

        for i in range(3):
            cl1.add_to_trace(ClassifierTrace.BAD)
            cl2.add_to_trace(ClassifierTrace.BAD)

        population = ClassifiersList(*[cl1, cl2])

        # when
        new_cls = list(ll.specialize_condition(population))

        # then
        assert len(new_cls) == 2

        assert new_cls[0].condition == Condition("01#1")
        assert new_cls[1].condition == Condition("11#1")

        assert all(cl.action == 0 for cl in new_cls)

        assert new_cls[0].effect == Effect("1122")
        assert new_cls[1].effect == Effect("#122")

        assert all(len(cl.trace) == 0 for cl in new_cls)

    def test_should_specialize(self, cfg, ll):
        # given
        # g1 - cl1, cl2
        # c1 don't have fully oscillating trace - group is skipped
        cl1 = Classifier(condition="#1#1", action=0, effect="1#22", cfg=cfg)
        cl2 = Classifier(condition="#1#1", action=0, effect="1#22", cfg=cfg)

        # g2 - cl3, cl4
        # classifiers eligible for specialization
        cl3 = Classifier(condition="#1#1", action=1, effect="1#22", cfg=cfg)
        cl4 = Classifier(condition="#1#1", action=1, effect="1#22", cfg=cfg)

        # g3 - cl5, cl6
        # should form a group when EIS in condition are different
        cl5 = Classifier(condition="#1#1", action=2, effect="1#22", cfg=cfg)
        cl6 = Classifier(condition="#1#1", action=2, effect="1#22", cfg=cfg)

        population = ClassifiersList(*[cl1, cl2, cl3, cl4, cl5, cl6])

        # fill in relevant classifier differences
        for cl in population:
            cl.add_to_trace(ClassifierTrace.BAD)
            cl.add_to_trace(ClassifierTrace.GOOD)
            cl.add_to_trace(ClassifierTrace.BAD)
            cl.add_to_trace(ClassifierTrace.GOOD)
            cl.add_to_trace(ClassifierTrace.BAD)

        del cl1.trace[-1]  # remove last trace
        cl5.condition[0].eis = 0.2
        cl6.condition[2].eis = 0.3

        # when
        ll.specialize(population)

        # then
        assert len(population) == 6
        assert cl1 in population
        assert cl2 in population
        assert cl3 not in population
        assert cl4 not in population
        assert cl5 not in population
        assert cl6 not in population

