import pytest

from lcs import Perception
from lcs.agents.racs import Configuration, Condition
from lcs.representations import UBR
from lcs.representations.utils import cover_ratio
from lcs.representations.RealValueEncoder import RealValueEncoder


class TestCondition:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder=RealValueEncoder(4))

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
        ([UBR(7, 7), UBR(4, 4)], 0.0625),
        ([UBR(7, 8), UBR(4, 5)], 0.125),
        ([UBR(2, 8), UBR(4, 10)], 0.4375),
    ])
    def test_should_calculate_cover_ratio(
            self, _condition, _covered_pct, cfg):
        cond = Condition(_condition, cfg=cfg)
        assert cover_ratio(cond, cfg.encoder) == _covered_pct

    @pytest.mark.parametrize("_condition, _perception, _result", [
        ([UBR(0, 15), UBR(0, 15)], [0.2, 0.4], True),
        ([UBR(0, 15), UBR(0, 2)], [0.5, 0.5], False),
        ([UBR(8, 8), UBR(10, 10)], [0.5, 0.65], True)
    ])
    def test_should_match_perception(
            self, _condition, _perception, _result, cfg):

        # given
        cond = Condition(_condition, cfg=cfg)
        p0 = Perception(_perception, oktypes=(float,))

        # then
        assert cond.does_match(p0) == _result

    @pytest.mark.parametrize("_cond1, _cond2, _result", [
        ([UBR(0, 15), UBR(0, 15)], [UBR(2, 4), UBR(5, 10)], True),
        ([UBR(6, 10), UBR(0, 15)], [UBR(2, 4), UBR(5, 10)], False),
        ([UBR(0, 15), UBR(4, 10)], [UBR(2, 4), UBR(6, 12)], False),
        ([UBR(2, 4), UBR(5, 5)], [UBR(2, 4), UBR(5, 5)], True),
    ])
    def test_should_subsume_condition(self, _cond1, _cond2, _result, cfg):
        # given
        cond1 = Condition(_cond1, cfg=cfg)
        cond2 = Condition(_cond2, cfg=cfg)

        # then
        assert cond1.subsumes(cond2) == _result

    @pytest.mark.parametrize("_cond, _result", [
        ([UBR(0, 15), UBR(0, 7)], 'OOOOOOOOOo|OOOOo.....')
    ])
    def test_should_visualize(self, _cond, _result, cfg):
        assert repr(Condition(_cond, cfg=cfg)) == _result
