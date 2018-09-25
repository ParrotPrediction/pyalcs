import pytest

from lcs.agents.racs import Classifier, Configuration, Condition, Effect
from lcs.agents.racs.components.genetic_algorithm import mutate
from lcs.representations import UBR
from lcs.representations.RealValueEncoder import RealValueEncoder


class TestGeneticAlgorithm:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder=RealValueEncoder(4))

    @pytest.mark.parametrize("_cond, _effect", [
        ([UBR(2, 5), UBR(5, 10)], [UBR(2, 5), UBR(5, 10)]),
        ([UBR(5, 2), UBR(10, 5)], [UBR(5, 2), UBR(10, 5)]),
        ([UBR(2, 2), UBR(5, 5)], [UBR(2, 2), UBR(5, 5)]),
        ([UBR(0, 15), UBR(0, 15)], [UBR(0, 15), UBR(0, 15)]),
    ])
    def test_aggressive_mutation(self, _cond, _effect, cfg):
        # given
        condition = Condition(_cond, cfg)
        effect = Effect(_effect, cfg)
        cl = Classifier(condition=condition, effect=effect, cfg=cfg)
        mu = 1.0

        # when
        mutate(cl, cfg.encoder.range, mu)

        # then
        for idx, (c, e) in enumerate(zip(cl.condition, cl.effect)):
            assert c.lower_bound <= condition[idx].lower_bound
            assert c.upper_bound >= condition[idx].upper_bound
            assert e.lower_bound <= effect[idx].lower_bound
            assert e.upper_bound >= effect[idx].upper_bound

    @pytest.mark.parametrize("_cond, _effect", [
        ([UBR(2, 5), UBR(5, 10)], [UBR(3, 6), UBR(1, 1)]),
    ])
    def test_disabled_mutation(self, _cond, _effect, cfg):
        # given
        condition = Condition(_cond, cfg)
        effect = Effect(_effect, cfg)
        cl = Classifier(condition=condition, effect=effect, cfg=cfg)
        mu = 0.0

        # when
        mutate(cl, cfg.encoder.range, mu)

        # then
        for idx, (c, e) in enumerate(zip(cl.condition, cl.effect)):
            assert c.lower_bound == condition[idx].lower_bound
            assert c.upper_bound == condition[idx].upper_bound
            assert e.lower_bound == effect[idx].lower_bound
            assert e.upper_bound == effect[idx].upper_bound
