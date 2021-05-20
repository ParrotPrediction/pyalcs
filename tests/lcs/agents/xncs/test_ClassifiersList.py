import pytest

from lcs import Perception
from lcs.agents.xncs import Configuration, Classifier
# TODO: Find the typo that makes it so you cannot import from .xncs
from lcs.agents.xncs.ClassifiersList import ClassifiersList
from lcs.agents.xcs import Condition

class TestClassifiersList:

    @pytest.fixture
    def cfg(self):
        return Configuration(number_of_actions=4)

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
    
    def test_init(self, cfg):
        cll = ClassifiersList(cfg)
        assert len(cll) == 0
        assert id(cll.cfg) == id(cfg)
        
    def test_covering(self, cfg):
        classifiers_list = ClassifiersList(cfg)
        covering_cl = classifiers_list.generate_covering_classifier(Perception("1111"), 0, 0)
        assert covering_cl.does_match(Perception("1111"))
        assert classifiers_list.generate_covering_classifier("1111", 0, 0).action == 0
        assert len(covering_cl.effect) == len(covering_cl.condition)

    def test_match_set(self, classifiers_list_diff_actions):
        assert len(classifiers_list_diff_actions.generate_match_set(Perception("1100"), 1)) == 4
        match_set = classifiers_list_diff_actions.generate_match_set(Perception("1111"), 1)
        assert len(match_set) == 4
        assert len(classifiers_list_diff_actions) == 8

    def test_action_set(self, cfg, classifiers_list_diff_actions):
        action_set = classifiers_list_diff_actions.generate_action_set(0)
        assert len(action_set) == 1
        assert action_set[0].action == 0
        action_set[0].action = 1
        assert classifiers_list_diff_actions[0].action == 1

    def test_return_fittest(self, classifiers_list_diff_actions):
        classifiers_list_diff_actions[2].prediction = 20
        classifiers_list_diff_actions[2].fitness = 20
        assert id(classifiers_list_diff_actions.fittest_classifier) == id(classifiers_list_diff_actions[2])
