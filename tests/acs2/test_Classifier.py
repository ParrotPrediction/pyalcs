import pytest

from alcs import Perception
from alcs.acs2 import ACS2Configuration, Classifier, Condition, Effect
from .randommock import RandomMock


class TestClassifier:

    @pytest.fixture
    def cfg(self):
        return ACS2Configuration(8, 8)

    def test_equality(self, cfg):
        # given
        cl = Classifier(action=1, numerosity=2, cfg=cfg)

        # when & then
        assert Classifier(action=1, numerosity=2, cfg=cfg) == cl

    def test_is_equally_general(self, cfg):
        c1 = Classifier(Condition('1#######', cfg), cfg=cfg)

        assert c1.is_equally_general(
            Classifier(Condition('1#######', cfg), cfg=cfg)) is True

        assert c1.is_equally_general(
            Classifier(Condition('0#######', cfg), cfg=cfg)) is True

        assert c1.is_equally_general(
            Classifier(Condition('#0######', cfg), cfg=cfg)) is True

        assert c1.is_equally_general(
            Classifier(Condition('#01#####', cfg), cfg=cfg)) is False

        assert c1.is_equally_general(
            Classifier(Condition('########', cfg), cfg=cfg)) is False

    def test_mutate_1(self, cfg):
        # given
        cls = Classifier(Condition('##011###', cfg), cfg=cfg)
        s = cfg.mu * 0.5  # less then MU
        b = 1 - (1 - cfg.mu) * 0.5  # more then MU

        # when
        cls.mutate(randomfunc=RandomMock([s, b, b]))

        # then
        assert Condition('###11###', cfg) == cls.condition

    def test_mutate_2(self, cfg):
        # given
        cls = Classifier(Condition('##011###', cfg), cfg=cfg)
        s = cfg.mu * 0.5  # less then MU
        b = 1 - (1 - cfg.mu) * 0.5  # more then MU

        # when
        cls.mutate(randomfunc=RandomMock([b, b, s]))

        # then
        assert Condition('##01####', cfg) == cls.condition

    def test_should_calculate_fitness(self, cfg):
        # given
        cls = Classifier(reward=0.25, cfg=cfg)

        # then
        assert 0.125 == cls.fitness

    def test_should_anticipate_change(self, cfg):
        # given
        cls = Classifier(cfg=cfg)
        assert cls.does_anticipate_change() is False

        # when
        cls.effect[1] = '1'

        # then
        assert cls.does_anticipate_change() is True

    def test_should_anticipate_correctly(self, cfg):
        # given
        cls = Classifier(
            effect=Effect('#1####0#', cfg),
            cfg=cfg)
        p0 = Perception('00001111')
        p1 = Perception('01001101')

        # then
        assert cls.does_anticipate_correctly(p0, p1) is True

    def test_should_calculate_specificity_1(self, cfg):
        cls = Classifier(cfg=cfg)
        assert 0 == cls.specificity

    def test_should_calculate_specificity_2(self, cfg):
        # given
        cls = Classifier(
            condition=Condition('#1#01#0#', cfg),
            cfg=cfg)

        # then
        assert 0.5 == cls.specificity

    def test_should_calculate_specificity_3(self, cfg):
        # given
        cls = Classifier(
            condition=Condition('11101001', cfg),
            cfg=cfg)

        # then
        assert 1 == cls.specificity

    def test_should_be_considered_as_reliable_1(self, cfg):
        # given
        cls = Classifier(quality=0.89, cfg=cfg)

        # then
        assert cls.is_reliable() is False

    def test_should_be_considered_as_reliable_2(self, cfg):
        # given
        cls = Classifier(quality=0.91, cfg=cfg)

        # then
        assert cls.is_reliable() is True

    def test_should_be_considered_as_inadequate_1(self, cfg):
        # given
        cls = Classifier(quality=0.50, cfg=cfg)

        # then
        assert cls.is_reliable() is False

    def test_should_be_considered_as_inadequate_2(self, cfg):
        # given
        cls = Classifier(quality=0.09, cfg=cfg)

        # then
        assert cls.is_inadequate() is True

    def test_should_update_reward(self, cfg):
        # given
        cls = Classifier(cfg=cfg)

        # when
        cls.update_reward(1000)

        # then
        assert 50.475 == cls.r

    def test_should_update_intermediate_reward(self, cfg):
        # given
        cls = Classifier(cfg=cfg)

        # when
        cls.update_intermediate_reward(1000)

        # then
        assert 50.0 == cls.ir

    def test_should_increase_experience(self, cfg):
        # given
        cls = Classifier(experience=5, cfg=cfg)

        # when
        cls.increase_experience()

        # then
        assert 6 == cls.exp

    def test_should_increase_quality(self, cfg):
        # given
        cls = Classifier(quality=0.5, cfg=cfg)

        # when
        cls.increase_quality()

        # then
        assert 0.525 == cls.q

    def test_should_decrease_quality(self, cfg):
        # given
        cls = Classifier(quality=0.47, cfg=cfg)

        # when
        cls.decrease_quality()

        # then
        assert abs(0.45 - cls.q) < 0.01

    def test_should_cover_triple(self, cfg):
        # given
        action_no = 2
        time = 123
        p0 = Perception('01001101')
        p1 = Perception('00011111')

        # when
        new_cl = Classifier.cover_triple(p0, action_no, p1, time, cfg)

        # then
        assert Condition('#1#0##0#', cfg) == new_cl.condition
        assert 2 == new_cl.action
        assert Effect('#0#1##1#', cfg) == new_cl.effect
        assert 0.5 == new_cl.q
        assert 0.5 == new_cl.r
        assert 0 == new_cl.ir
        assert 0 == new_cl.tav
        assert time == new_cl.tga
        assert time == new_cl.talp
        assert 1 == new_cl.num
        assert 1 == new_cl.exp
