import pytest

from lcs import Perception
from lcs.agents.acs import Classifier, Configuration, Condition, Effect


class TestClassifier:

    @pytest.fixture
    def cfg(self):
        return Configuration(4, 4)

    def test_should_create_general_classifier(self, cfg):
        cl = Classifier.general(action=1, cfg=cfg)

        assert cl.condition == Condition("####")
        assert cl.action == 1
        assert cl.condition == Effect("####")

    @pytest.mark.parametrize("_c, _e, _result", [
        ("####", "####", True),
        ("####", "###1", False),
        ("###1", "####", False),
        ("###1", "###1", False)
    ])
    def test_distinguish_general_classifier(self, _c, _e, _result, cfg):
        cl = Classifier(
            condition=Condition(_c),
            effect=Effect(_e),
            cfg=cfg
        )

        assert cl.is_general() == _result

    def test_should_decrease_quality(self, cfg):
        cl = Classifier(cfg=cfg)

        cl.decrease_quality()

        assert cl.q == 0.475

    def test_should_increase_quality(self, cfg):
        cl = Classifier(cfg=cfg)

        cl.increase_quality()

        assert cl.q == 0.525

    @pytest.mark.parametrize("_c, _e, _p0, _p1, _result", [
        ('000#', '000#', "0000", "0001", True),
        ('0#0#', '0#0#', "0000", "0101", True),
        ('####', '####', "0000", "1111", True),
        ('0001', '000#', "0000", "0001", False),
        ('000#', '0001', "0000", "0001", False),
        ('010#', '0#0#', "0000", "0101", False),
        ('0#0#', '0#01', "0000", "0101", False),
    ])
    def test_should_detect_correctable_classifier(
            self, _c, _e, _p0, _p1, _result, cfg):

        cl = Classifier(condition=_c, effect=_e, cfg=cfg)
        p0 = Perception(_p0)
        p1 = Perception(_p1)

        assert cl.can_be_corrected(p0, p1) is _result

    def test_should_construct_correct_classifier(self, cfg):
        cl = Classifier(condition="000#", effect="000#", quality=0.2, cfg=cfg)
        p0 = Perception("0000")
        p1 = Perception("0001")
        assert cl.does_match(p0)

        new_cl = Classifier.build_corrected(cl, p0, p1)

        assert new_cl is not cl
        assert new_cl.condition is not cl.condition
        assert new_cl.effect is not cl.effect
        assert new_cl.condition == Condition("0000")
        assert new_cl.action == cl.action
        assert new_cl.effect == Effect("0001")
        assert new_cl.q == 0.5
        assert new_cl.does_match(p0)
