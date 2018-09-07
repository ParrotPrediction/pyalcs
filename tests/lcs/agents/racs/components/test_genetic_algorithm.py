import pytest

from lcs.agents.racs import Classifier, Configuration, Condition
from lcs.agents.racs.components.genetic_algorithm import mutate
from lcs.representations import UBR


class TestGeneticAlgorithm:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder_bits=4)

    @pytest.mark.parametrize("_cond", [
        ([UBR(2, 5), UBR(5, 10)]),
        ([UBR(5, 2), UBR(10, 5)]),
        ([UBR(2, 2), UBR(5, 5)]),
        ([UBR(0, 15), UBR(0, 15)]),
    ])
    def test_aggressive_mutation(self, _cond, cfg):
        # given
        condition = Condition(_cond, cfg)
        cl = Classifier(condition=condition, cfg=cfg)
        mu = 1.0

        # when
        mutate(cl, cfg.encoder.range, mu)

        # then
        for idx, ubr in enumerate(cl.condition):
            assert ubr.lower_bound <= condition[idx].lower_bound
            assert ubr.upper_bound >= condition[idx].upper_bound

    @pytest.mark.parametrize("_cond", [
        ([UBR(2, 5), UBR(5, 10)]),
    ])
    def test_disabled_mutation(self, _cond, cfg):
        # given
        condition = Condition(_cond, cfg)
        cl = Classifier(condition=condition, cfg=cfg)
        mu = 0.0

        # when
        mutate(cl, cfg.encoder.range, mu)

        # then
        for idx, ubr in enumerate(cl.condition):
            assert ubr.lower_bound == condition[idx].lower_bound
            assert ubr.upper_bound == condition[idx].upper_bound
