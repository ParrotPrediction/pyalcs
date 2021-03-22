import pytest
from copy import copy

from lcs import Perception
from lcs.agents.xcs import Configuration, Condition, Classifier, ClassifiersList, XCS, GeneticAlgorithm
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

    def test_mutation(self, cfg):
        cfg.mutation_chance = 0
        cl = Classifier(cfg, Condition("####"), 0, 0)
        GeneticAlgorithm._apply_mutation(cl, cfg, Perception("1111"))
        assert cl.action == Classifier(cfg, Condition("1111"), 0, 0).action
        assert cl.does_match(Condition("1111"))
        cfg.mutation_chance = 1
        cl = Classifier(cfg, Condition("1111"), 0, 0)
        GeneticAlgorithm._apply_mutation(cl, cfg, Perception("1111"))
        assert cl.is_more_general(Classifier(cfg, Condition("1111"), 0, 0))

    @pytest.mark.parametrize("cond1, cond2, x, y, end_cond1, end_cond2", [
        ("11111", "#####", 0, 5, "#####", "11111"),
        ("11111", "000", 0, 5, "00011", "111"),
        ("11111", "#####", 1, 4, "1###1", "#111#")
    ])
    def test_crossover(self, cfg, cond1, cond2, x, y, end_cond1, end_cond2):
        cl1 = Classifier(cfg, Condition(cond1), 0, 0)
        cl2 = Classifier(cfg, Condition(cond2), 1, 0)
        GeneticAlgorithm._apply_crossover_in_area(cl1, cl2, x, y)
        assert cl1.condition == Condition(end_cond1)
        assert cl2.condition == Condition(end_cond2)
        # Just to check for errors
        GeneticAlgorithm._apply_crossover(cl1, cl2)


    @pytest.mark.parametrize("chi", [
        1,
        0
    ])
    # only tests for errors and types
    def test_run_ga(self, cfg, classifiers_list_diff_actions, chi):
        cfg.do_GA_subsumption = True
        xcs = XCS(cfg, classifiers_list_diff_actions)
        action_set = xcs.population.generate_action_set(0)
        cfg.chi = chi  # do perform crossover
        cfg.do_GA_subsumption = False  # do not perform subsumption
        GeneticAlgorithm.run_ga(xcs.population, action_set, Perception("0000"), 100000, cfg)
        assert xcs.population[0].time_stamp != 0
        assert xcs.population.numerosity > 4

    def test_simple_q_learning(self, cfg, classifiers_list_diff_actions):
        reward = simple_q_learning(0, 0, cfg.learning_rate, cfg.gamma, 0)
        assert reward == 0
        reward = simple_q_learning(0, 10, cfg.learning_rate, cfg.gamma, 0)
        assert reward > 0
        new_reward = simple_q_learning(reward, 0, cfg.learning_rate, cfg.gamma, 0)
        assert new_reward < reward

    def test_make_children(self, cfg, classifiers_list_diff_actions):
        child1, child2 = GeneticAlgorithm._make_children(
            classifiers_list_diff_actions[0],
            classifiers_list_diff_actions[1]
        )
        assert child1.numerosity == 1
        assert child2.numerosity == 1
        assert child1.experience == 0
        assert child2.experience == 0
        assert id(child1) != id(classifiers_list_diff_actions[0])
        assert id(child2) != id(classifiers_list_diff_actions[0])
        assert id(child1) != id(classifiers_list_diff_actions[1])
        assert id(child2) != id(classifiers_list_diff_actions[1])

