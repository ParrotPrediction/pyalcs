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

    # TODO: You can test if correct element was deleted
    # TODO: Include random state to manipulate the rnd and see the above TODO
    def test_deletion(self, cfg: Configuration):
        classifiers_list = ClassifiersList(cfg)
        for i in range(cfg.n + 1):
            classifiers_list.insert_in_population(
                Classifier(cfg, Condition("1111"), 0, 0)
            )
        assert sum(cl.numerosity for cl in classifiers_list) >= cfg.n
        classifiers_list.delete_from_population()
        assert sum(cl.numerosity for cl in classifiers_list) <= cfg.n

    def test_match_set(self, cfg):
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 3, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1111"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1111"), 1, 0))
        assert len(classifiers_list.form_match_set(Perception("1100"), 1)) == 4
        assert len(classifiers_list.form_match_set(Perception("1110"), 1)) == 4

    def test_action_set(self, cfg):
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 3, 0))
        action_set = classifiers_list.form_action_set(0)
        assert len(action_set) == 1
        action_set[0].action = 1
        assert classifiers_list[0].action == 1

