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

    @pytest.mark.parametrize("_init_cond, _other_cond, _result_cond", [
        ([UBR(0, 10), UBR(5, 2)],
         [UBR(0, 15), UBR(0, 15)],
         [UBR(0, 10), UBR(2, 5)]),
        ([UBR(0, 10), UBR(5, 2)],
         [UBR(3, 12), UBR(0, 15)],
         [UBR(3, 12), UBR(2, 5)])
    ])
    def test_should_specialize_with_condition(
            self, _init_cond, _other_cond, _result_cond, cfg):

        # given
        cond = Condition(_init_cond, cfg)
        other = Condition(_other_cond, cfg)

        # when
        cond.specialize_with_condition(other)

        # then
        assert cond == Condition(_result_cond, cfg)

    @pytest.mark.parametrize("_condition, _idx, _generalized", [
        ([UBR(1, 4), UBR(5, 7)], 0, [UBR(0, 15), UBR(5, 7)]),
        ([UBR(1, 4), UBR(5, 7)], 1, [UBR(1, 4), UBR(0, 15)])
    ])
    def test_generalize(self, _condition, _idx, _generalized, cfg):
        # given
        cond = Condition(_condition, cfg)

        # when
        cond.generalize(_idx)

        # then
        assert cond == Condition(_generalized, cfg)

    @pytest.mark.parametrize("_condition, _spec_before, _spec_after", [
        ([UBR(2, 6), UBR(7, 2)], 2, 1),
        ([UBR(2, 6), UBR(0, 15)], 1, 0),
        ([UBR(0, 15), UBR(0, 15)], 0, 0),
    ])
    def test_should_generalize_specific_attributes_randomly(
            self, _condition, _spec_before, _spec_after, cfg):

        # given
        condition = Condition(_condition, cfg)
        assert condition.specificity == _spec_before

        # when
        condition.generalize_specific_attribute_randomly()

        # then
        assert condition.specificity == _spec_after

    @pytest.mark.parametrize("_condition, _specificity", [
        ([UBR(0, 15), UBR(0, 15)], 0),
        ([UBR(0, 15), UBR(2, 15)], 1),
        ([UBR(5, 15), UBR(2, 12)], 2)
    ])
    def test_should_count_specificity(self, _condition, _specificity, cfg):
        cond = Condition(_condition, cfg=cfg)
        assert cond.specificity == _specificity

    @pytest.mark.parametrize("_condition, _covered_pct", [
        ([UBR(0, 15), UBR(0, 15)], 1.0),
        ([UBR(7, 7), UBR(4, 4)], 0.0),
        ([UBR(7, 8), UBR(4, 5)], 0.06),
        ([UBR(2, 8), UBR(4, 10)], 0.4),
    ])
    def test_should_calculate_cover_ratio(
            self, _condition, _covered_pct, cfg):
        cond = Condition(_condition, cfg=cfg)
        assert abs(cond.cover_ratio - _covered_pct) < 0.05

    @pytest.mark.parametrize("_condition, _perception, _result", [
        ([UBR(0, 15), UBR(0, 15)], [0.2, 0.4], True),
        ([UBR(0, 15), UBR(0, 2)], [0.5, 0.5], False),
        ([UBR(7, 7), UBR(10, 10)], [0.5, 0.7], True)
    ])
    def test_should_match_perception(
            self, _condition, _perception, _result, cfg):

        # given
        cond = Condition(_condition, cfg=cfg)
        p0 = Perception(_perception, oktypes=(float,))

        # then
        assert cond.does_match(p0) == _result
