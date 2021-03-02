import pytest

from lcs import Perception
from lcs.agents.macs.macs import Configuration, LatentLearning, \
    ClassifiersList, Classifier


class TestLatentLearning:
    P0 = Perception('1000')
    ACTION = 0
    P1 = Perception('1001')

    SEED = 31337

    @pytest.fixture
    def cfg(self):
        return Configuration(4, 2)

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

    def test_should_suppress_classifier_when_evaluating(
        self, population, ll, cfg):
        # given
        [cl1, cl2, cl3, cl4] = population
        cl1.b = 4

        # when
        ll.evaluate_classifiers(population, self.P0, self.ACTION, self.P1)

        # then
        assert len(population) == 3
        assert cl1 not in population
        self._assert_gb_metrics(cl2, 0, 0)
        self._assert_gb_metrics(cl3, 1, 0)
        self._assert_gb_metrics(cl4, 0, 0)

    @staticmethod
    def _assert_gb_metrics(cl: Classifier, ga, ba):
        assert cl.g == ga
        assert cl.b == ba
