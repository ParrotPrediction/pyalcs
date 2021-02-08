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

    def test_should_determine_specializable_classifier(self, cfg):
        # Correct case - trace full and oscillating
        cl1 = Classifier(cfg=cfg)
        cl1.add_to_trace(ClassifierTrace.BAD)
        cl1.add_to_trace(ClassifierTrace.BAD)
        cl1.add_to_trace(ClassifierTrace.BAD)
        cl1.add_to_trace(ClassifierTrace.GOOD)
        cl1.add_to_trace(ClassifierTrace.BAD)
        assert cl1.is_specializable() is True

        # Wrong case - trace not full and oscillating
        cl2 = Classifier(cfg=cfg)
        cl2.add_to_trace(ClassifierTrace.BAD)
        cl2.add_to_trace(ClassifierTrace.GOOD)
        cl2.add_to_trace(ClassifierTrace.BAD)
        assert cl2.is_specializable() is False

        # Wrong case - trace full not oscillating
        cl3 = Classifier(cfg=cfg)
        cl3.add_to_trace(ClassifierTrace.BAD)
        cl3.add_to_trace(ClassifierTrace.BAD)
        cl3.add_to_trace(ClassifierTrace.BAD)
        cl3.add_to_trace(ClassifierTrace.BAD)
        cl3.add_to_trace(ClassifierTrace.BAD)
        assert cl3.is_specializable() is False
