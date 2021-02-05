import pytest

from lcs.agents.yacs.yacs import Configuration, Classifier, ClassifierTrace


class TestClassifier:

    @pytest.fixture
    def cfg(self):
        return Configuration(2, 2, feature_possible_values=[2, 2])

    def test_should_add_to_trace(self, cfg):
        cls = Classifier(cfg=cfg)
        assert len(cls.trace) == 0

        cls.add_to_trace(ClassifierTrace.GOOD)
        assert len(cls.trace) == 1

        cls.add_to_trace(ClassifierTrace.BAD)
        cls.add_to_trace(ClassifierTrace.BAD)
        cls.add_to_trace(ClassifierTrace.BAD)
        cls.add_to_trace(ClassifierTrace.BAD)
        cls.add_to_trace(ClassifierTrace.BAD)
        assert len(cls.trace) == cfg.trace_length
        assert all([True for t in cls.trace if t == ClassifierTrace.BAD])
