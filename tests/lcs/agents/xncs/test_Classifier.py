import pytest

from lcs.agents.xncs import Classifier, Configuration


class TestClassifier:

    @pytest.fixture
    def cfg(self):
        return Configuration(lmc=2, lem=0.2, number_of_actions=4)

    def test_init(self, cfg):
        cl = Classifier(cfg)
        assert cl.cfg == cfg
        assert cl.effect is None

