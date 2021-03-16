import pytest

from lcs import Perception
from lcs.agents import PerceptionString
from lcs.agents.xcs import Classifier, Configuration, Condition


class TestClassifier:

    @pytest.fixture
    def cfg(self):
        return Configuration(number_of_actions=4)

    def test_classifier_default(self, cfg):
        cl = Classifier(cfg,
                        Condition("####"),
                        2,
                        8)
        assert cl.condition == Condition('####')
        assert cl.prediction == cfg.initial_error

    def test_subsumes(self, cfg: Configuration):
        assert Classifier(cfg, Condition("1100"), 0, 0).condition.subsumes("1100")
        assert Classifier(cfg, Condition("1100"), 0, 0).condition.subsumes(Perception("1100"))
        assert Classifier(cfg, Condition("1100"), 0, 0).condition.subsumes(Condition("1100"))
        assert Classifier(cfg, Condition("1100"), 0, 0).condition.subsumes(PerceptionString("1100"))
        assert Classifier(cfg, Condition("1100"), 0, 0).condition.subsumes("11##")
        assert not Classifier(cfg, Condition("1111"), 0, 0).condition.subsumes("0000")
        assert not Classifier(cfg, Condition("1100"), 0, 0).condition.subsumes("1111")
        assert not Classifier(cfg, Condition("11##"), 0, 0).condition.subsumes("10##")
        assert not Classifier(cfg, Condition("11##"), 0, 0).condition.subsumes("1000")

    def test_does_match(self, cfg: Configuration):
        assert Classifier(cfg, Condition("1100"), 0, 0).does_match("1100")
        assert Classifier(cfg, Condition("1100"), 0, 0).does_match(Perception("1100"))
        assert Classifier(cfg, Condition("1100"), 0, 0).does_match(Condition("1100"))
        assert Classifier(cfg, Condition("1100"), 0, 0).does_match(PerceptionString("1100"))
        assert Classifier(cfg, Condition("1100"), 0, 0).does_match("11##")
        assert not Classifier(cfg, Condition("1111"), 0, 0).does_match("0000")
        assert not Classifier(cfg, Condition("1100"), 0, 0).does_match("1111")
        assert not Classifier(cfg, Condition("11##"), 0, 0).does_match("10##")
        assert not Classifier(cfg, Condition("11##"), 0, 0).does_match("1000")

    def test_could_subsume(self, cfg: Configuration):
        cl = Classifier(cfg, Condition("1111"), 0, 0)
        assert not cl.could_subsume
        cl.experience = cfg.subsumption_threshold * 2
        cl.error = cfg.initial_error / 2
        assert cl.could_subsume

    def test_is_more_general(self, cfg: Configuration):
        assert Classifier(cfg, Condition("####"), 0, 0).is_more_general(
            Classifier(cfg, Condition("1111"), 0, 0)
        )
        assert not Classifier(cfg, Condition("1111"), 0, 0).is_more_general(
            Classifier(cfg, Condition("1111"), 0, 0)
        )
        assert not Classifier(cfg, Condition("1111"), 0, 0).is_more_general(
            Classifier(cfg, Condition("11##"), 0, 0)
        )
        assert not Classifier(cfg, Condition("###0"), 0, 0).is_more_general(
            Classifier(cfg, Condition("1111"), 0, 0)
        )

    def test_does_subsume(self, cfg: Configuration):
        cl = Classifier(cfg, Condition("11##"), 0, 0)
        assert not cl.does_subsume(Classifier(cfg, Condition("1111"), 0, 0))
        cl.experience = cfg.subsumption_threshold * 2
        cl.error = cfg.initial_error / 2
        assert cl.does_subsume(Classifier(cfg, Condition("1111"), 0, 0))
        assert not cl.does_subsume(Classifier(cfg, Condition("1111"), 1, 0))
        assert not cl.does_subsume(Classifier(cfg, Condition("0011"), 1, 0))
