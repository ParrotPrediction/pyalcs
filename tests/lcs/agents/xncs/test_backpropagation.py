import pytest

from lcs.agents.xncs import Backpropagation, Configuration, Classifier, Effect, ClassifiersList
from lcs.agents.xcs import Condition


class TestBackpropagation:


    @pytest.fixture
    def cfg(self):
        return Configuration(lmc=4, lem=20, number_of_actions=4)

    @pytest.fixture
    def situation(self):
        return "1100"

    @pytest.fixture
    def classifiers_list_diff_actions(self, cfg, situation):
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 0, 0, Effect(situation)))
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 1, 0, Effect(situation)))
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 2, 0, Effect(situation)))
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 3, 0, Effect(situation)))
        return classifiers_list

    def test_update_effect(self, cfg, classifiers_list_diff_actions):
        classifiers_list_diff_actions[0].fitness = 1000
        classifiers_list_diff_actions[1].fitness = 0
        classifiers_list_diff_actions[2].fitness = 0
        bp = Backpropagation(cfg, 0.5)
        bp.update_effect(classifiers_list_diff_actions)
        assert classifiers_list_diff_actions[1].effect == classifiers_list_diff_actions[0].effect

    def test_insertion(self, cfg, classifiers_list_diff_actions):
        bp = Backpropagation(cfg, 0.5)
        bp.update_cycle(classifiers_list_diff_actions, Effect("1111"))
        assert len(bp.classifiers_for_update) == 4
        bp.update_cycle(classifiers_list_diff_actions, Effect("1111"))
        assert len(bp.classifiers_for_update) == 4

    def test_deletion(self, cfg, classifiers_list_diff_actions):
        bp = Backpropagation(cfg, 0.5)
        bp.update_cycle(classifiers_list_diff_actions, Effect("1111"))
        assert len(bp.classifiers_for_update) == 4
        bp.classifiers_for_update[1][2] = 1
        bp.update_cycle(classifiers_list_diff_actions, Effect("1111"))
        assert len(bp.classifiers_for_update) == 3

    def test_errors(self, cfg: Configuration, classifiers_list_diff_actions):
        bp = Backpropagation(cfg, 0.5)
        bp.update_cycle(classifiers_list_diff_actions, Effect("1111"))
        assert classifiers_list_diff_actions[0].error != cfg.initial_error

