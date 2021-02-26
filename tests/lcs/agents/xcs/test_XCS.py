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

    @pytest.fixture
    def xcs(self, cfg):
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 3, 0))
        xcs = XCS(cfg)
        return xcs

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

    def test_mutation(self, cfg):
        cfg.mu = 0
        cl = Classifier(cfg, Condition("1111"), 0, 0)
        xcs = XCS(cfg)
        xcs.apply_mutation(cl, Perception("0000"))
        assert cl == Classifier(cfg, Condition("1111"), 0, 0)
        cfg.mu = 2
        xcs.apply_mutation(cl, Perception("0000"))
        assert cl != Classifier(cfg, Condition("1111"), 0, 0)

    # only tests for errors and types
    def test_crossover(self, cfg):
        cl1 = Classifier(cfg, Condition("1111"), 0, 0)
        cl2 = Classifier(cfg, Condition("0000"), 1, 0)
        xcs = XCS(cfg)
        xcs.apply_crossover(cl1, cl2)

    # only tests for errors and types
    def test_run_GA(self, cfg):
        cfg.do_GA_subsumption = True
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 3, 0))
        xcs = XCS(cfg, classifiers_list)
        action_set = xcs.population.form_action_set(0)
        xcs.run_ga(action_set, Perception("0000"), 1)

    # only tests for errors and types
    def test_do_action_set_subsumption(self, xcs):
        action_set = xcs.population.form_action_set(0)
        xcs.do_action_set_subsumption(action_set)

    def test_update_set(self, cfg):
        cfg.do_GA_subsumption = True
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 3, 0))
        xcs = XCS(cfg, classifiers_list)
        action_set = xcs.population.form_action_set(0)
        cl = copy.copy(action_set[0])
        cfg.beta = 1
        xcs._update_set(action_set, 0.2)
        assert classifiers_list[0].experience > 0
        assert classifiers_list[0].prediction != cl.prediction
        assert classifiers_list[0].error != cl.error
        cfg.beta = 0.000000001
        cl = copy.copy(action_set[0])
        xcs._update_set(action_set, 0.2)
        assert classifiers_list[0].experience > 1

