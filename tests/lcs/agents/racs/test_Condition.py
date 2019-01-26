import pytest

from lcs import Perception
from lcs.agents.racs import Configuration, Condition
from lcs.representations import Interval


class TestCondition:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2)

    def test_should_create_generic_condition(self, cfg):
        # when
        cond = Condition.generic(cfg)

        # then
        assert len(cond) == cfg.classifier_length
        for allele in cond:
            assert allele == cfg.classifier_wildcard

    @pytest.mark.parametrize("_init_cond, _other_cond, _result_cond", [
        ([Interval(0., .8), Interval(.5, .2)],
         [Interval(0., 1.), Interval(0., 1.)],
         [Interval(0., .8), Interval(.2, .5)]),
        ([Interval(0., .8), Interval(.5, .2)],
         [Interval(.2, .9), Interval(0., 1.)],
         [Interval(.2, .9), Interval(.2, .5)])
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
        ([Interval(.1, .4), Interval(.5, .7)], 0,
         [Interval(0., 1.), Interval(.5, .7)]),
        ([Interval(.1, .4), Interval(.5, .7)], 1,
         [Interval(.1, .4), Interval(0., 1.)])
    ])
    def test_generalize(self, _condition, _idx, _generalized, cfg):
        # given
        cond = Condition(_condition, cfg)

        # when
        cond.generalize(_idx)

        # then
        assert cond == Condition(_generalized, cfg)

    @pytest.mark.parametrize("_condition, _spec_before, _spec_after", [
        ([Interval(.2, .6), Interval(.7, .2)], 2, 1),
        ([Interval(.2, .6), Interval(0., 1.)], 1, 0),
        ([Interval(0., 1.), Interval(0., 1.)], 0, 0),
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
        ([Interval(0., 1.), Interval(0., 1.)], 0),
        ([Interval(0., 1.), Interval(.2, 1.)], 1),
        ([Interval(.5, 1.), Interval(.2, .8)], 2)
    ])
    def test_should_count_specificity(self, _condition, _specificity, cfg):
        cond = Condition(_condition, cfg=cfg)
        assert cond.specificity == _specificity

    @pytest.mark.parametrize("_condition, _covered_pct", [
        ([Interval(0., 1.), Interval(0., 1.)], 1.0),
        ([Interval(0., .5), Interval(.5, 1.)], 0.5),
        ([Interval(.7, .8), Interval(.4, .5)], 0.1),
        ([Interval(.2, .8), Interval(.7, .8)], 0.35),
    ])
    def test_should_calculate_cover_ratio(
            self, _condition, _covered_pct, cfg):
        cond = Condition(_condition, cfg=cfg)
        assert abs(cond.cover_ratio - _covered_pct) < 0.00001

    @pytest.mark.parametrize("_condition, _perception, _result", [
        ([Interval(0., 1.), Interval(0., 1.)], [0.2, 0.4], True),
        ([Interval(0., 1.), Interval(0., .2)], [0.5, 0.5], False),
        ([Interval(.5, .5), Interval(.65, .65)], [0.5, 0.65], True),
        ([Interval(.49, .51), Interval(.64, .65)], [0.5, 0.65], True),
    ])
    def test_should_match_perception(
            self, _condition, _perception, _result, cfg):

        # given
        cond = Condition(_condition, cfg=cfg)
        p0 = Perception(_perception, oktypes=(float,))

        # then
        assert cond.does_match(p0) == _result

    @pytest.mark.parametrize("_cond1, _cond2, _result", [
        ([Interval(0., 1.), Interval(0., 1.)],
         [Interval(.2, .4), Interval(.5, .8)], True),
        ([Interval(.7, .9), Interval(0., 1.)],
         [Interval(.2, .4), Interval(.5, .8)], False),
        ([Interval(0., 1.), Interval(.4, .6)],
         [Interval(.2, .4), Interval(.5, .9)], False),
        ([Interval(.2, .4), Interval(.5, .5)],
         [Interval(.2, .4), Interval(.5, .5)], True),
        ([Interval(.2, .4), Interval(.5, .5)],
         [Interval(.5, .5), Interval(.2, .4)], False),
    ])
    def test_should_subsume_condition(self, _cond1, _cond2, _result, cfg):
        # given
        cond1 = Condition(_cond1, cfg=cfg)
        cond2 = Condition(_cond2, cfg=cfg)

        # then
        assert cond1.subsumes(cond2) == _result

    # @pytest.mark.parametrize("_cond, _result", [
    #     ([UBR(0, 15), UBR(0, 7)], 'OOOOOOOOOO|OOOOO.....')
    # ])
    # def test_should_visualize(self, _cond, _result, cfg):
    #     assert repr(Condition(_cond, cfg=cfg)) == _result
