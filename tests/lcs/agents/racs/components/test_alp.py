import pytest
import random

from lcs import Perception
from lcs.agents.racs import Configuration, Condition, Effect, Classifier
from lcs.agents.racs.components.alp import unexpected_case
from lcs.representations import UBR


class TestALP:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder_bits=4)

    def test_should_handle_unexpected_case_1(self, cfg):
        # given
        p0 = Perception([.5, .5], oktypes=(float,))
        p1 = Perception([.5, .5], oktypes=(float,))
        # Effect is all pass-through. Can be specialized.
        effect = Effect([UBR(0, 16), UBR(0, 16)], cfg=cfg)
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
        assert child.condition == Condition([UBR(0, 16), UBR(0, 16)], cfg=cfg)
        assert child.effect == Effect([UBR(0, 16), UBR(0, 16)], cfg=cfg)

    def test_should_handle_unexpected_case_2(self, cfg):
        # given
        p0 = Perception([.5, .5], oktypes=(float,))
        p1 = Perception([.5, .5], oktypes=(float,))
        # Effect is not specializable
        effect = Effect([UBR(0, 16), UBR(2, 4)], cfg=cfg)
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
        effect = Effect([UBR(0, 16), UBR(10, 14)], cfg=cfg)
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
        assert child.condition == Condition([UBR(0, 16), UBR(8, 8)], cfg=cfg)
        assert child.effect == Effect([UBR(0, 16), UBR(12, 12)], cfg=cfg)
