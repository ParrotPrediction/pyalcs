import pytest

from lcs import Perception
from lcs.agents.racs import Configuration, Condition
from lcs.representations import UBR


class TestCondition:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder_bits=4)

    def test_should_create_generic_condition(self, cfg):
        # when
        cond = Condition.generic(cfg)

        # then
        assert len(cond) == cfg.classifier_length
        for allele in cond:
            assert allele == cfg.classifier_wildcard

    @pytest.mark.parametrize("_condition, _specificity", [
        ([UBR(0, 16), UBR(0, 16)], 0),
        ([UBR(0, 16), UBR(2, 16)], 1),
        ([UBR(5, 16), UBR(2, 12)], 2)
    ])
    def test_should_count_specificity(self, _condition, _specificity, cfg):
        cond = Condition(_condition, cfg=cfg)
        assert cond.specificity == _specificity

    @pytest.mark.parametrize("_condition, _perception, _result", [
        ([UBR(0, 16), UBR(0, 16)], [0.2, 0.4], True),
        ([UBR(0, 16), UBR(0, 2)], [0.5, 0.5], False),
        ([UBR(8, 8), UBR(11, 11)], [0.5, 0.7], True)
    ])
    def test_should_match_perception(
            self, _condition, _perception, _result, cfg):

        # given
        cond = Condition(_condition, cfg=cfg)
        p0 = Perception(_perception, oktypes=(float,))

        # then
        assert cond.does_match(p0) == _result
