import pytest

from lcs import Perception
from lcs.agents.racs import Configuration, Condition, Effect, Classifier
from lcs.representations import UBR


class TestClassifier:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder_bits=4)

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

    def test_should_anticipate_change_1(self, cfg):
        # given
        p0 = Perception([0.5, 0.5], oktypes=(float,))
        p1 = Perception([0.5, 0.5], oktypes=(float,))
        # Classifier with default pass-through effect
        c = Classifier(cfg=cfg)

        # then
        assert c.does_anticipate_correctly(p0, p1) is True

    def test_should_anticipate_change_2(self, cfg):
        # given
        effect = Effect([cfg.classifier_wildcard, UBR(10, 12)],
                        cfg=cfg)
        p0 = Perception([0.5, 0.5], oktypes=(float,))
        p1 = Perception([0.5, 0.5], oktypes=(float,))
        c = Classifier(effect=effect, cfg=cfg)

        # then
        assert c.does_anticipate_correctly(p0, p1) is False

    def test_should_anticipate_change_3(self, cfg):
        # given
        effect = Effect([UBR(0, 4), UBR(10, 12)],
                        cfg=cfg)
        p0 = Perception([0.8, 0.8], oktypes=(float,))  # encoded [12, 12]
        p1 = Perception([0.2, 0.7], oktypes=(float,))  # encoded [3, 11]
        c = Classifier(effect=effect, cfg=cfg)

        # then
        assert c.does_anticipate_correctly(p0, p1) is True

    def test_should_decrease_quality(self, cfg):
        # given
        cl = Classifier(cfg=cfg)
        assert cl.q == 0.5

        # when
        cl.decrease_quality()

        # then
        assert cl.q == 0.475
