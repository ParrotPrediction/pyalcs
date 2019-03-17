import random

import pytest

from lcs import Perception
from lcs.agents.racs import Configuration, Condition, Effect, Classifier
from lcs.agents.racs.components.alp \
    import cover, expected_case, unexpected_case
from lcs.representations import Interval


class TestALP:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2)

    def test_should_handle_expected_case_1(self, cfg):
        # given
        p0 = Perception([.5, .5], oktypes=(float,))
        q = random.random()
        cl = Classifier(quality=q, cfg=cfg)
        time = random.randint(0, 1000)

        # when
        child = expected_case(cl, p0, time)

        # then
        # classifier is not marked - no child should be generated
        assert child is None
        assert cl.q > q

    def test_should_handle_expected_case_2(self, cfg):
        # given
        p0 = Perception([.5, .5], oktypes=(float,))
        q = random.random()
        cl = Classifier(quality=q, cfg=cfg)
        cl.mark[0].add(.505)
        cl.mark[1].add(.505)
        time = random.randint(0, 1000)

        # when
        child = expected_case(cl, p0, time)

        # then
        # there are no differences
        assert child is None
        assert cl.q > q

    def test_should_handle_expected_case_3(self, cfg):
        # given
        p0 = Perception([.5, .5], oktypes=(float,))
        q = 0.4
        cl = Classifier(quality=q, cfg=cfg)
        cl.mark[0].add(.2)
        time = random.randint(0, 1000)

        # when
        child = expected_case(cl, p0, time)

        # then
        assert child is not None
        assert child.condition == Condition(
            [Interval(.5, .5), Interval(0., 1.)], cfg)
        assert child.q == 0.5

    def test_should_handle_unexpected_case_1(self, cfg):
        # given
        p0 = Perception([.5, .5], oktypes=(float,))
        p1 = Perception([.5, .5], oktypes=(float,))
        # Effect is all pass-through. Can be specialized.
        effect = Effect([Interval(0., 1.), Interval(0., 1.)], cfg=cfg)
        quality = .4
        time = random.randint(0, 1000)
        cl = Classifier(effect=effect, quality=quality, cfg=cfg)

        # when
        child = unexpected_case(cl, p0, p1, time)

        # then
        assert cl.q < quality
        assert cl.is_marked() is True
        assert child
        assert child.q == .5
        assert child.talp == time
        # There is no change in perception so the child condition
        # and effect should stay the same.
        assert child.condition == Condition(
            [Interval(0., 1.), Interval(0., 1.)], cfg=cfg)
        assert child.effect == Effect(
            [Interval(0., 1.), Interval(0., 1.)], cfg=cfg)

    def test_should_handle_unexpected_case_2(self, cfg):
        # given
        p0 = Perception([.5, .5], oktypes=(float,))
        p1 = Perception([.5, .5], oktypes=(float,))
        # Effect is not specializable
        effect = Effect([Interval(0., 1.), Interval(.2, .4)], cfg=cfg)
        quality = random.random()
        time = random.randint(0, 1000)
        cl = Classifier(effect=effect, quality=quality, cfg=cfg)

        # when
        child = unexpected_case(cl, p0, p1, time)

        # then
        assert cl.q < quality
        assert cl.is_marked() is True
        # We cannot generate child from non specializable parent
        assert child is None

    def test_should_handle_unexpected_case_3(self, cfg):
        # given
        p0 = Perception([.5, .5], oktypes=(float,))
        p1 = Perception([.5, .8], oktypes=(float,))
        # Second effect attribute is specializable
        effect = Effect([Interval(0., 1.), Interval(.8, .9)], cfg=cfg)
        quality = 0.4
        time = random.randint(0, 1000)
        cl = Classifier(effect=effect, quality=quality, cfg=cfg)

        # when
        child = unexpected_case(cl, p0, p1, time)

        # then
        assert cl.q < quality
        assert cl.is_marked() is True

        assert child is not None
        assert child.is_marked() is False
        assert child.q == .5
        assert child.condition == Condition(
            [Interval(0., 1.), Interval(0., 1.)], cfg=cfg)
        assert child.effect == Effect(
            [Interval(0., 1.), Interval(.8, .9)], cfg=cfg)

    @pytest.mark.parametrize("_p0, _p1, _child_cond, _child_effect", [
        ([.5, .5], [.5, .5],
         [Interval(0., 1.), Interval(0., 1.)],
         [Interval(0., 1.), Interval(0., 1.)]),
        ([.4, .5], [.9, .5],
         [Interval(.4, .4), Interval(0., 1.)],
         [Interval(.9, .9), Interval(0., 1.)]),
    ])
    def test_should_create_new_classifier_with_covering(
            self, _p0, _p1, _child_cond, _child_effect, cfg):

        # given
        cfg.cover_noise = 0.0

        p0 = Perception(_p0, oktypes=(float,))
        p1 = Perception(_p1, oktypes=(float,))
        action = random.randint(0, cfg.number_of_possible_actions)
        time = random.randint(0, 100)

        # when
        new_cl = cover(p0, action, p1, time, cfg)

        # then
        assert new_cl is not None
        assert new_cl.condition == Condition(_child_cond, cfg)
        assert new_cl.action == action
        assert new_cl.effect == Effect(_child_effect, cfg)
        assert new_cl.q == .5
        assert new_cl.r == 0
        assert new_cl.ir == 0
        assert new_cl.tav == 0
        assert new_cl.tga == time
        assert new_cl.talp == time
        assert new_cl.num == 1
        assert new_cl.exp == 0
