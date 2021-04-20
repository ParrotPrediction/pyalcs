import pytest
from copy import copy

from lcs import Perception
from lcs.agents.xncs import XNCS, Classifier, Configuration, Backpropagation
from lcs.agents.xcs import ClassifiersList, Condition


class TestXNCS:

    @pytest.fixture
    def cfg(self):
        return Configuration(lmc=2, lem=0.2, number_of_actions=4)

    @pytest.fixture
    def situation(self):
        return "1100"

    @pytest.fixture
    def classifiers_list_diff_actions(self, cfg, situation):
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 3, 0))
        return classifiers_list

    def test_init(self, cfg, classifiers_list_diff_actions):
        xncs = XNCS(cfg, classifiers_list_diff_actions)
        assert type(xncs.back_propagation) == Backpropagation
        assert id(xncs.cfg) == id(cfg)
        assert len(xncs.population) == 4

    def test_distribute_and_update(self, cfg, classifiers_list_diff_actions):
        xncs = XNCS(cfg, classifiers_list_diff_actions)
        xncs._distribute_and_update(classifiers_list_diff_actions, "1100", 0.1)
        assert len(xncs.back_propagation.update_vectors) == 4
