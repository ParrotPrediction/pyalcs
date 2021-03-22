import pytest

from lcs.agents.macs.macs import ClassifiersList, Configuration, Classifier


class TestClassifiersList:

    @pytest.fixture
    def cfg(self):
        feature_vals = {'0', '1'}
        return Configuration(2, 2, feature_possible_values=[feature_vals] * 2)

    def test_should_initialize(self, cfg):
        cls = ClassifiersList()
        assert len(cls) == 0

        cls = ClassifiersList(*[
            Classifier(condition='00', action=0, effect='1?', cfg=cfg)
        ])
        assert len(cls) == 1
