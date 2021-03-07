import pytest

from lcs import Perception
from lcs.agents.macs.macs import Configuration, LatentLearning, \
    ClassifiersList, Classifier, Effect, Condition


class TestLatentLearning:
    P0 = Perception('1000')
    ACTION = 0
    P1 = Perception('1001')

    SEED = 31337

    @pytest.fixture
    def cfg(self):
        return Configuration(4, 2, [2, 2, 2, 2])

    @pytest.fixture
    def ll(self, cfg):
        return LatentLearning(cfg)

    @pytest.fixture
    def population(self, cfg):
        # wrong anticipation
        cl1 = Classifier(condition='1###', action=0, effect='??1?', cfg=cfg)

        # wrong action
        cl2 = Classifier(condition='1###', action=1, effect='?001', cfg=cfg)

        # correct anticipation
        cl3 = Classifier(condition='1###', action=0, effect='???1', cfg=cfg)

        # wrong condition
        cl4 = Classifier(condition='0###', action=0, effect='1???', cfg=cfg)

        return ClassifiersList(*[cl1, cl2, cl3, cl4])

    def test_should_evaluate_classifiers(self, population, ll, cfg):
        # given
        for cl in population:
            assert cl.g == 0
            assert cl.b == 0

        # when
        ll.evaluate_classifiers(population, self.P0, self.ACTION, self.P1)

        # then
        assert len(population) == 4
        [cl1, cl2, cl3, cl4] = population

        self._assert_gb_metrics(cl1, 0, 1)
        self._assert_gb_metrics(cl2, 0, 0)
        self._assert_gb_metrics(cl3, 1, 0)
        self._assert_gb_metrics(cl4, 0, 0)

    def test_should_suppress_inaccurate_classifier(
        self, population, ll, cfg):
        # given
        [cl1, cl2, cl3, cl4] = population
        cl1.b = 5

        # when
        ll.select_accurate(population)

        # then
        assert len(population) == 3
        assert cl1 not in population

    def test_should_generate_using_mutspec(self, ll, cfg):
        # given
        cl = Classifier(condition='####', action=0, effect='???1', cfg=cfg)
        feature_idx = 0

        # when
        new_cls = list(ll.mutspec(cl, feature_idx))

        # then
        assert len(new_cls) == 2
        assert all(cl.action == 0 for cl in new_cls)
        assert all(cl.effect == Effect('???1') for cl in new_cls)
        assert new_cls[0].condition == Condition('0###')
        assert new_cls[1].condition == Condition('1###')

    def test_should_specialize_conditions(self, population, ll, cfg):
        # given
        [cl1, cl2, cl3, cl4] = population
        cl3.g = 3
        cl3.b = 3
        perceptions = {self.P0}

        # when
        ll.specialize_conditions(population, perceptions)

        # then
        assert len(population) == 5
        assert any(cl.condition == Condition('10##') for cl in population)

    def test_should_update_ig_estimates(self, population, ll, cfg):
        # given
        assert all(cl.condition.ig == [0.5, 0.5, 0.5, 0.5] for cl in population)

        # when
        # cl4 is eligible for generalization
        ll.evaluate_generalization_estimates(
            population, self.P0, self.ACTION, self.P1)

        # then
        [cl1, cl2, cl3, cl4] = population
        assert all(
            cl.condition.ig == [0.5, 0.5, 0.5, 0.5] for cl in [cl1, cl2, cl3])
        assert cl4.condition.ig == [0.55, 0.5, 0.5, 0.5]

    @staticmethod
    def _assert_gb_metrics(cl: Classifier, ga, ba):
        assert cl.g == ga
        assert cl.b == ba
