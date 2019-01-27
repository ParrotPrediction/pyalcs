import pytest

from lcs.agents.racs import Configuration, ClassifierList, \
    Classifier, Condition
from lcs.agents.racs.metrics import count_averaged_regions
from lcs.representations import Interval


class TestMetrics:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2)

    def test_regions_averaging(self, cfg):
        # given
        cl1 = Classifier(
            condition=Condition([Interval(.2, .3), Interval(.4, .5)], cfg),
            cfg=cfg)
        cl2 = Classifier(
            condition=Condition([Interval(0., .3), Interval(.4, .9)], cfg),
            cfg=cfg)
        cl3 = Classifier(
            condition=Condition([Interval(.1, .3), Interval(.4, 1.)], cfg),
            cfg=cfg)
        cl4 = Classifier(
            condition=Condition([Interval(0., .9), Interval(0., 1.)], cfg),
            cfg=cfg)
        population = ClassifierList(*[cl1, cl2, cl3, cl4])

        # when
        result = count_averaged_regions(population)

        # then
        assert type(result) is dict
        assert result == {1: 0.5, 2: 0.25, 3: 0.125, 4: 0.125}
