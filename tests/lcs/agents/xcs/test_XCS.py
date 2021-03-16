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
        return Configuration(number_of_actions=number_of_actions, do_action_set_subsumption=False)

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
        assert xcs.population[0].fitness != cfg.initial_fitness
        assert classifiers_list[0].fitness != cfg.initial_fitness

    def test_mutation(self, cfg):
        cfg.mutation_chance = 0
        cl = Classifier(cfg, Condition("####"), 0, 0)
        xcs = XCS(cfg)
        xcs.apply_mutation(cl, Perception("1111"))
        assert cl.action == Classifier(cfg, Condition("1111"), 0, 0).action
        assert cl.does_match(Condition("1111"))
        cfg.mutation_chance = 1
        cl = Classifier(cfg, Condition("1111"), 0, 0)
        xcs.apply_mutation(cl, Perception("1111"))
        assert cl.is_more_general(Classifier(cfg, Condition("1111"), 0, 0))

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
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1111"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1111"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1111"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1111"), 3, 0))
        xcs = XCS(cfg, classifiers_list)
        action_set = xcs.population.form_action_set(0)

        cfg.chi = 0  # do perform crossover
        cfg.do_GA_subsumption = False  # do not perform subsumption
        xcs.run_ga(action_set, Perception("0000"), 100000)
        assert xcs.population[0].time_stamp != 0
        assert xcs.population.numerosity() > 4

        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1111"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1111"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1111"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1111"), 3, 0))
        xcs = XCS(cfg, classifiers_list)
        action_set = xcs.population.form_action_set(0)

        cfg.chi = 0  # do perform crossover
        cfg.do_GA_subsumption = True  # do not perform subsumption
        xcs.run_ga(action_set, Perception("0000"), 100000)
        assert xcs.population[0].time_stamp != 0
        assert xcs.population.numerosity() > 4


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
        xcs.update_set(action_set, 0.2)
        assert classifiers_list[0].experience > 0
        assert classifiers_list[0].prediction != cl.prediction
        assert classifiers_list[0].error != cl.error
        cfg.beta = 0.000000001
        cl = copy.copy(action_set[0])
        xcs.update_set(action_set, 0.2)
        assert classifiers_list[0].experience > 1

