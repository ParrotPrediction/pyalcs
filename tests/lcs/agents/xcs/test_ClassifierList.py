import pytest
import random

from lcs import Perception
from lcs.agents.xcs import Configuration, Condition, Classifier, ClassifiersList


class TestClassifiersList:

    @pytest.fixture
    def cfg(self):
        return Configuration(4, 4)

    def test_init(self, cfg):
        assert len(ClassifiersList(cfg)) == 0

    def test_insert_population(self, cfg):
        classifiers_list = ClassifiersList(cfg)
        assert len(classifiers_list) == 0
        classifiers_list.insert_in_population(
            Classifier(cfg, Condition("1111"), 0, 0))
        assert len(classifiers_list) == 1
        classifiers_list.insert_in_population(
            Classifier(cfg, Condition("1111"), 0, 1))
        assert len(classifiers_list) == 1
        classifiers_list.insert_in_population(
            Classifier(cfg, Condition("1111"), 1, 2))
        assert len(classifiers_list) == 2
        assert classifiers_list[0].numerosity == 2
        assert classifiers_list[1].numerosity == 1

    def test_covering(self, cfg):
        classifiers_list = ClassifiersList(cfg)
        assert Classifier(cfg, Condition("1111"), 0, 0) ==\
            classifiers_list.generate_covering_classifier(Perception("1111"), 0, 0)
        assert Classifier(cfg, Condition("1111"), 0, 0) ==\
            classifiers_list.generate_covering_classifier("1111", 0, 0)

    # TODO: Finish this test
    def test_deletion(self, cfg: Configuration):
        classifiers_list = ClassifiersList(cfg)
        for i in range(cfg.n + 1):
            classifiers_list.insert_in_population(
                Classifier(cfg, Condition("1111"), 0, 0)
            )
        assert sum(cl.numerosity for cl in classifiers_list) >= cfg.n
        classifiers_list.delete_from_population()
        assert sum(cl.numerosity for cl in classifiers_list) <= cfg.n
