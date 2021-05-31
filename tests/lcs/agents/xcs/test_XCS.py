import pytest
from copy import copy

from lcs import Perception
from lcs.agents.xcs import Configuration, Condition, Classifier, ClassifiersList, XCS
from lcs.strategies.reinforcement_learning import simple_q_learning

class TestXCS:

    @pytest.fixture
    def number_of_actions(self):
        return 4

    @pytest.fixture
    def cfg(self, number_of_actions):
        return Configuration(number_of_actions=number_of_actions, do_action_set_subsumption=False)

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

    @pytest.fixture
    def xcs(self, cfg, classifiers_list_diff_actions):
        xcs = XCS(cfg=cfg, population=classifiers_list_diff_actions)
        return xcs

    # it mostly tests if function will manage to run
    def test_select_action(self, xcs, classifiers_list_diff_actions):
        prediction_array = [0, 0, 1, 0]
        xcs.cfg.epsilon = -1
        action = xcs.select_action(prediction_array=prediction_array,
                                   match_set=classifiers_list_diff_actions)
        assert type(action) == int
        assert action == 2

        xcs.cfg.epsilon = 2
        action = xcs.select_action(prediction_array=prediction_array,
                                   match_set=classifiers_list_diff_actions)
        assert type(action) == int
        assert action < xcs.cfg.number_of_actions

    # only tests for errors and types
    # TODO: Do more tests here
    def test_do_action_set_subsumption(self, xcs):
        action_set = xcs.population.generate_action_set(0)
        xcs.do_action_set_subsumption(action_set)

    def test_distribute_and_update(self, cfg, situation, classifiers_list_diff_actions):
        xcs = XCS(cfg=cfg, population=classifiers_list_diff_actions)
        prediction_array = [1, 1, 1, 1]
        reward = 1

        xcs._distribute_and_update(None, situation, reward)
        for cl in xcs.get_population():
            assert cl.prediction == cfg.initial_prediction
            assert cl.fitness == cfg.initial_fitness
            assert cl.error == cfg.initial_error

        xcs._distribute_and_update(xcs.population, situation, reward)
        for cl in xcs.get_population():
            assert not cl.prediction == cfg.initial_prediction
            assert not cl.fitness == cfg.initial_fitness
            assert not cl.error == cfg.initial_error


