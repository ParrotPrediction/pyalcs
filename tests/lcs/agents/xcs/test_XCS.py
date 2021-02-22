import pytest
import random

from lcs import Perception
from lcs.agents.xcs import Configuration, Condition, Classifier, ClassifiersList, XCS


class TestXCS:

    @pytest.fixture
    def number_of_actions(self):
        return 4

    @pytest.fixture
    def cfg(self, number_of_actions):
        return Configuration(number_of_actions, 4)

    def test_prediction_array(self, cfg):
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 3, 0))
        xcs = XCS()
        prediction_array = xcs.generate_prediction_array(classifiers_list)
        assert len(classifiers_list) == len(prediction_array)

    # it mostly tests if function will manage to run
    def test_select_action(self, cfg, number_of_actions):
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 3, 0))
        xcs = XCS(cfg)
        prediction_array = xcs.generate_prediction_array(classifiers_list)
        action = xcs.select_action(prediction_array=prediction_array,
                                   match_set=classifiers_list)

        assert type(action) == int
        assert action < number_of_actions
