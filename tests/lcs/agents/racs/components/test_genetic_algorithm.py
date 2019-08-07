from copy import deepcopy

import numpy as np
import pytest

from lcs.agents.racs import Classifier, Configuration, Condition, Effect
from lcs.agents.racs.components.genetic_algorithm import mutate, crossover, \
    _flatten, _unflatten
from lcs.representations import Interval, FULL_INTERVAL


class TestGeneticAlgorithm:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2)

    @pytest.mark.parametrize("_cond, _effect", [
        ([Interval(.2, .5), Interval(.5, .7)],
         [Interval(.2, .5), Interval(.5, .7)]),
        ([Interval(.5, .2), Interval(.8, .5)],
         [Interval(.5, .2), Interval(.8, .5)])
    ])
    def test_if_elements_are_mutated(self, _cond, _effect, cfg):
        # given
        np.random.seed(12345)
        condition = Condition(_cond, cfg)
        effect = Effect(_effect, cfg)

        cfg.mutation_noise = 0.5  # strong noise mutation range
        mu = 1.0  # mutate every attribute

        cl = Classifier(
            condition=deepcopy(condition),
            effect=deepcopy(effect),
            cfg=cfg)

        # when
        mutate(cl, mu)

        # then
        for idx, (c, e) in enumerate(zip(cl.condition, cl.effect)):
            assert condition[idx] != c
            assert effect[idx] != e

    @pytest.mark.parametrize("_cond, _effect", [
        ([Interval(.2, .5), Interval(.5, 1.)],
         [Interval(.2, .5), Interval(.5, 1.)]),
        ([Interval(.5, .2), Interval(.8, .5)],
         [Interval(.5, .2), Interval(.8, .5)]),
        ([Interval(.2, .2), Interval(.3, .3)],
         [Interval(.2, .2), Interval(.3, .3)]),
        ([Interval(0., 1.), Interval(0., 1.)],
         [Interval(0., 1.), Interval(0., 1.)]),
    ])
    def test_if_range_is_not_exceeded_in_mutation(self, _cond, _effect, cfg):
        # given
        condition = Condition(_cond, cfg)
        effect = Effect(_effect, cfg)

        cfg.mutation_noise = 0.5  # strong noise mutation range
        mu = 1.0  # mutate every attribute

        cl = Classifier(
            condition=deepcopy(condition),
            effect=deepcopy(effect),
            cfg=cfg)

        # when
        mutate(cl, mu)

        # then
        for idx, (c, e) in enumerate(zip(cl.condition, cl.effect)):
            # assert if condition values are in ranges
            assert c.left >= FULL_INTERVAL.left
            assert c.right <= FULL_INTERVAL.right

            # assert if effect values are in ranges
            assert e.left >= FULL_INTERVAL.left
            assert e.right <= FULL_INTERVAL.right

    @pytest.mark.parametrize("_cond, _effect", [
        ([Interval(.2, .5), Interval(.5, .8)],
         [Interval(.2, .5), Interval(.5, .8)]),
    ])
    def test_disabled_mutation(self, _cond, _effect, cfg):
        # given
        condition = Condition(_cond, cfg)
        effect = Effect(_effect, cfg)
        cl = Classifier(
            condition=deepcopy(condition),
            effect=deepcopy(effect),
            cfg=cfg)
        mu = 0.0

        # when
        mutate(cl, mu)

        # then
        for idx, (c, e) in enumerate(zip(cl.condition, cl.effect)):
            assert c.left == condition[idx].left
            assert c.right == condition[idx].right
            assert e.left == effect[idx].left
            assert e.right == effect[idx].right

    def test_crossover(self, cfg):
        # given
        parent = Classifier(
            condition=Condition(
                [Interval(.1, .1), Interval(.1, .1), Interval(.1, .1)], cfg),
            effect=Effect(
                [Interval(.1, .1), Interval(.1, .1), Interval(.1, .1)], cfg),
            cfg=cfg)
        donor = Classifier(
            condition=Condition(
                [Interval(.2, .2), Interval(.2, .2), Interval(.2, .2)], cfg),
            effect=Effect(
                [Interval(.2, .2), Interval(.2, .2), Interval(.2, .2)], cfg),
            cfg=cfg)

        # when
        np.random.seed(12345)  # left: 3, right: 6
        crossover(parent, donor)

        # then
        assert parent.condition == \
            Condition(
                [Interval(.1, .1), Interval(.1, .2), Interval(.2, .2)], cfg)
        assert parent.effect == \
            Effect(
                [Interval(.1, .1), Interval(.1, .2), Interval(.2, .2)], cfg)
        assert donor.condition == \
            Condition(
                [Interval(.2, .2), Interval(.2, .1), Interval(.1, .1)], cfg)
        assert donor.effect == \
            Effect(
                [Interval(.2, .2), Interval(.2, .1), Interval(.1, .1)], cfg)

    @pytest.mark.parametrize("_cond, _result", [
        ([Interval(.1, .3), Interval(.2, .4)], [.1, .3, .2, .4])
    ])
    def test_should_flatten_condition(self, _cond, _result, cfg):
        assert _flatten(Condition(_cond, cfg=cfg)) == _result

    @pytest.mark.parametrize("_effect, _result", [
        ([Interval(.1, .3), Interval(.2, .4)], [.1, .3, .2, .4])
    ])
    def test_should_flatten_effect(self, _effect, _result, cfg):
        assert _flatten(Effect(_effect, cfg=cfg)) == _result

    @pytest.mark.parametrize("_flat, _result", [
        ([.1, .3], [Interval(.1, .3)]),
        ([.1, .3, .2, .4], [Interval(.1, .3), Interval(.2, .4)])
    ])
    def test_should_unflatten(self, _flat, _result):
        assert _unflatten(_flat) == _result
