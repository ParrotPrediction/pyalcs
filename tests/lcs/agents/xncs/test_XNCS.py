import pytest
from copy import copy

from lcs import Perception
from lcs.agents.xncs import XNCS, Classifier, Configuration, Backpropagation, ClassifiersList, Effect
from lcs.agents.xcs import Condition


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
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 0, 0, Effect(situation)))
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 1, 0, Effect(situation)))
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 2, 0, Effect(situation)))
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 3, 0, Effect(situation)))
        return classifiers_list

    def test_init(self, cfg, classifiers_list_diff_actions):
        xncs = XNCS(cfg, classifiers_list_diff_actions)
        assert type(xncs.back_propagation) == Backpropagation
        assert xncs.back_propagation is not None
        assert id(xncs.cfg) == id(cfg)
        assert len(xncs.population) == 4
        assert isinstance(classifiers_list_diff_actions, ClassifiersList)
        assert isinstance(xncs.population, ClassifiersList)

    def test_init_no_pop(self, cfg):
        xncs = XNCS(cfg)
        assert isinstance(xncs.population, ClassifiersList)

    def test_distribute_and_update(self, cfg: Configuration,
                                   classifiers_list_diff_actions,
                                   situation):
        xncs = XNCS(cfg, classifiers_list_diff_actions)
        action_set = xncs.population.generate_action_set(0)
        assert action_set is not None
        assert action_set.fittest_classifier is not None
        xncs._distribute_and_update(action_set, "####", "####", 0.1)
        # update should happen because effect matched inserted vector
        assert len(xncs.back_propagation.classifiers_for_update) == 1

    def test_distribute_and_update_diff(self, cfg: Configuration,
                                        classifiers_list_diff_actions,
                                        situation):
        xncs = XNCS(cfg, classifiers_list_diff_actions)
        action_set = xncs.population.generate_action_set(0)
        assert action_set is not None
        assert action_set.fittest_classifier is not None
        xncs._distribute_and_update(action_set, "####", "####", 0.1)

    def test_correct_type_population(self, cfg):
        xncs = XNCS(cfg)
        assert isinstance(xncs.population, ClassifiersList)

    def test_correct_type_classifier(self, cfg):
        xncs = XNCS(cfg)
        xncs.population.insert_in_population(
            Classifier(cfg, Condition("1100"), 0, 0)
        )
        assert isinstance(xncs.population[0], Classifier)
        assert xncs.population[0].effect is None
