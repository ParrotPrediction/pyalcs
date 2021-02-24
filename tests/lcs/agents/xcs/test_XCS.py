import pytest
import copy

from lcs import Perception
from lcs.agents.xcs import Configuration, Condition, Classifier, ClassifiersList, XCS


class TestXCS:

    @pytest.fixture
    def number_of_actions(self):
        return 4

    @pytest.fixture
    def cfg(self, number_of_actions):
        return Configuration(number_of_actions, 4, do_action_set_subsumption=False)

    def test_prediction_array(self, cfg):
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 3, 0))
        xcs = XCS(cfg)
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
        xcs.cfg.p_exp = 0.999999 # I know that is lazy
        assert type(action) == int
        assert action < number_of_actions

        xcs.cfg.p_exp = 0
        action = xcs.select_action(prediction_array=prediction_array,
                                   match_set=classifiers_list)
        assert type(action) == int
        assert action < number_of_actions

    def test_update_fitness(self, cfg):
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 3, 0))
        xcs = XCS(cfg, classifiers_list)
        action_set = xcs.population.form_action_set(0)
        xcs.update_fitness(action_set)
        assert xcs.population[0].fitness != cfg.f_i
        assert classifiers_list[0].fitness != cfg.f_i

    # TODO: test update_set
    # TODO: test do_action_set_subsumbtion
