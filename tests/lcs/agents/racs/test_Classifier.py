import random

import pytest

from lcs import Perception
from lcs.agents.racs import Configuration, Condition, Effect, Classifier
from lcs.representations import UBR
from lcs.representations.RealValueEncoder import RealValueEncoder


class TestClassifier:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder=RealValueEncoder(4))

    def test_should_initialize_without_arguments(self, cfg):
        # when
        c = Classifier(cfg=cfg)

        # then
        assert c.condition == Condition.generic(cfg=cfg)
        assert c.action is None
        assert c.effect == Effect.pass_through(cfg=cfg)
        assert c.exp == 1
        assert c.talp is None
        assert c.tav == 0.0

    def test_should_detect_identical_classifier(self, cfg):
        cl_1 = Classifier(
            condition=Condition([UBR(0, 1), UBR(0, 2)], cfg=cfg),
            action=1,
            effect=Effect([UBR(2, 3), UBR(4, 5)], cfg=cfg),
            cfg=cfg)

        cl_2 = Classifier(
            condition=Condition([UBR(0, 1), UBR(0, 2)], cfg=cfg),
            action=1,
            effect=Effect([UBR(2, 3), UBR(4, 5)], cfg=cfg),
            cfg=cfg)

        assert cl_1 == cl_2

    @pytest.mark.parametrize("_q, _r, _fitness", [
        (0.0, 0.0, 0.0),
        (0.3, 0.5, 0.15),
        (1.0, 1.0, 1.0),
    ])
    def test_should_calculate_fitness(self, _q, _r, _fitness, cfg):
        assert Classifier(quality=_q, reward=_r, cfg=cfg).fitness == _fitness

    @pytest.mark.parametrize("_effect, _p0, _p1, _result", [
        # Classifier with default pass-through effect
        (None, [0.5, 0.5], [0.5, 0.5], True),
        ([UBR(0, 15), UBR(10, 12)], [0.5, 0.5], [0.5, 0.5], False),
        ([UBR(0, 4), UBR(10, 12)], [0.8, 0.8], [0.2, 0.7], True),
        # second perception attribute is unchanged - should be a wildcard
        ([UBR(0, 4), UBR(10, 12)], [0.8, 0.8], [0.2, 0.8], False),
    ])
    def test_should_anticipate_change(self, _effect, _p0, _p1, _result, cfg):
        # given
        p0 = Perception(_p0, oktypes=(float,))
        p1 = Perception(_p1, oktypes=(float,))

        c = Classifier(effect=_effect, cfg=cfg)

        # then
        assert c.does_anticipate_correctly(p0, p1) is _result

    @pytest.mark.parametrize("_q, _reliable", [
        (.5, False),
        (.1, False),
        (.9, False),
        (.91, True),
    ])
    def test_should_detect_reliable(self, _q, _reliable, cfg):
        # given
        cl = Classifier(quality=_q, cfg=cfg)

        # then
        assert cl.is_reliable() is _reliable

    @pytest.mark.parametrize("_q, _inadequate", [
        (.5, False),
        (.1, False),
        (.09, True),
    ])
    def test_should_detect_inadequate(self, _q, _inadequate, cfg):
        # given
        cl = Classifier(quality=_q, cfg=cfg)

        # then
        assert cl.is_inadequate() is _inadequate

    def test_should_increase_quality(self, cfg):
        # given
        cl = Classifier(cfg=cfg)
        assert cl.q == 0.5

        # when
        cl.increase_quality()

        # then
        assert cl.q == 0.525

    def test_should_decrease_quality(self, cfg):
        # given
        cl = Classifier(cfg=cfg)
        assert cl.q == 0.5

        # when
        cl.decrease_quality()

        # then
        assert cl.q == 0.475

    @pytest.mark.parametrize("_condition, _effect, _sua", [
        ([UBR(4, 15), UBR(2, 15)], [UBR(0, 15), UBR(0, 15)], 2),
        ([UBR(4, 15), UBR(0, 15)], [UBR(0, 15), UBR(0, 15)], 1),
        ([UBR(0, 15), UBR(0, 15)], [UBR(0, 15), UBR(0, 15)], 0),
        ([UBR(4, 15), UBR(0, 15)], [UBR(0, 15), UBR(5, 15)], 1),
        ([UBR(4, 15), UBR(6, 15)], [UBR(4, 15), UBR(6, 15)], 0),
    ])
    def test_should_count_specified_unchanging_attributes(
            self, _condition, _effect, _sua, cfg):

        # given
        cl = Classifier(condition=Condition(_condition, cfg),
                        effect=Effect(_effect, cfg),
                        cfg=cfg)

        # then
        assert len(cl.specified_unchanging_attributes) == _sua

    def test_should_create_copy(self, cfg):
        # given
        operation_time = random.randint(0, 100)
        condition = Condition([self._random_ubr(), self._random_ubr()],
                              cfg=cfg)
        action = random.randint(0, 2)
        effect = Effect([self._random_ubr(), self._random_ubr()], cfg=cfg)

        cl = Classifier(condition, action, effect,
                        quality=random.random(),
                        reward=random.random(),
                        intermediate_reward=random.random(),
                        cfg=cfg)
        # when
        copied_cl = Classifier.copy_from(cl, operation_time)

        # then
        assert cl is not copied_cl
        assert cl.condition == copied_cl.condition
        assert cl.condition is not copied_cl.condition
        assert cl.action == copied_cl.action
        assert cl.effect == copied_cl.effect
        assert cl.effect is not copied_cl.effect
        assert copied_cl.is_marked() is False
        assert cl.r == copied_cl.r
        assert cl.q == copied_cl.q
        assert operation_time == copied_cl.tga
        assert operation_time == copied_cl.talp

    def test_should_specialize(self, cfg):
        # given
        p0 = Perception([random.random()] * 2, oktypes=(float,))
        p1 = Perception([random.random()] * 2, oktypes=(float,))
        cl = Classifier(cfg=cfg)

        # when
        cl.specialize(p0, p1)

        # then
        for condition_ubr, effect_ubr in zip(cl.condition, cl.effect):
            assert condition_ubr.lower_bound == condition_ubr.upper_bound
            assert effect_ubr.lower_bound == effect_ubr.upper_bound

    @pytest.mark.parametrize("_condition, _effect, _soa_before, _soa_after", [
        ([UBR(4, 15), UBR(2, 15)], [UBR(0, 15), UBR(0, 15)], 2, 1),
        ([UBR(4, 15), UBR(0, 15)], [UBR(0, 15), UBR(0, 15)], 1, 0),
        ([UBR(0, 15), UBR(0, 15)], [UBR(0, 15), UBR(0, 15)], 0, 0),
    ])
    def test_should_generalize_randomly_unchanging_condition_attribute(
            self, _condition, _effect, _soa_before, _soa_after, cfg):

        # given
        condition = Condition(_condition, cfg)
        effect = Effect(_effect, cfg)
        cl = Classifier(condition=condition, effect=effect, cfg=cfg)
        assert len(cl.specified_unchanging_attributes) == _soa_before

        # when
        cl.generalize_unchanging_condition_attribute()

        # then
        assert (len(cl.specified_unchanging_attributes)) == _soa_after

    @pytest.mark.parametrize("_c1, _c2, _result", [
        ([UBR(4, 6), UBR(1, 5)], [UBR(4, 6), UBR(1, 4)], True),
        ([UBR(4, 6), UBR(1, 5)], [UBR(4, 6), UBR(1, 6)], False),
        # The same classifiers
        ([UBR(4, 6), UBR(1, 5)], [UBR(4, 6), UBR(1, 5)], False)
    ])
    def test_should_find_more_general(self, _c1, _c2, _result, cfg):
        # given
        cl1 = Classifier(condition=Condition(_c1, cfg), cfg=cfg)
        cl2 = Classifier(condition=Condition(_c2, cfg), cfg=cfg)

        # then
        assert cl1.is_more_general(cl2) is _result

    @staticmethod
    def _random_ubr(lower=0, upper=15):
        return UBR(random.randint(lower, upper), random.randint(lower, upper))
