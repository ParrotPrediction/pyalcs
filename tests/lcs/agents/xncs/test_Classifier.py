import pytest

from lcs.agents.xncs import Classifier, Configuration, Effect
from lcs.agents.xcs import Condition


class TestClassifier:

    @pytest.fixture
    def cfg(self):
        return Configuration(lmc=2, lem=0.2, number_of_actions=4)

    def test_init(self, cfg):
        cl = Classifier(condition='####', cfg=cfg)
        assert cl.cfg == cfg
        assert cl.effect is None

    @pytest.mark.parametrize("cond1, cond2, act1, act2, result, ef1, ef2", [
        ("1111", "1111", 1, 1, True, "1111", "1111"),
        ("#100", "#100", 0, 0, True, "1111", "1111"),

        ("1111", "1111", 1, 0, False, "1111", "1111"),
        ("1111", "1111", 0, 1, False, "1111", "1111"),
        ("1111", "1111", 1, 2, False, "1111", "1111"),
        ("1111", "1111", 1, 3, False, "1111", "1111"),
        ("1111", "1111", 2, 1, False, "1111", "1111"),
        ("1111", "1111", 3, 1, False, "1111", "1111"),

        ("1100", "1111", 1, 1, False, "1111", "1111"),
        ("1111", "1100", 1, 1, False, "1111", "1111"),

        ("1111", "####", 1, 1, False, "1111", "1111"),
        ("1111", "11##", 1, 1, False, "1111", "1111"),
        ("##11", "11##", 1, 1, False, "1111", "1111"),
        ("##11", "1111", 1, 1, False, "1111", "1111"),

        ("1111", "11", 1, 1, False, "1111", "1111"),
        ("11", "1111", 1, 1, False, "1111", "1111"),

        ("1111", "1111", 1, 1, False, "1111", "1100"),
        ("1111", "1111", 1, 1, False, "1100", "1111"),

    ])
    def test_equals(self, cfg, cond1, cond2, act1, act2, result, ef1, ef2):
        cl1 = Classifier(cfg=cfg, condition=Condition(cond1), action=act1, time_stamp=0)
        cl2 = Classifier(cfg=cfg, condition=Condition(cond2), action=act2, time_stamp=0)
        assert cl2.effect is None
        assert cl1.effect is None
        cl1.effect = Effect(ef1)
        cl2.effect = Effect(ef2)
        assert result == (cl1 == cl2)
