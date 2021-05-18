import pytest
from copy import copy

from lcs import Perception
from lcs.agents.xcs import Configuration, Condition, Classifier, ClassifiersList, XCS, GeneticAlgorithm
from lcs.strategies.reinforcement_learning import simple_q_learning

class TestGA:

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

    @pytest.fixture
    def ga(self, classifiers_list_diff_actions, cfg):
        ga = GeneticAlgorithm(classifiers_list_diff_actions, cfg)
        return ga

    def test_mutation(self, cfg, ga):
        cfg.mutation_chance = 0
        cl = Classifier(cfg, Condition("####"), 0, 0)
        ga._apply_mutation(cl, cfg, Perception("1111"))
        assert cl.action == Classifier(cfg, Condition("1111"), 0, 0).action
        assert cl.does_match(Condition("1111"))
        cfg.mutation_chance = 1
        cl = Classifier(cfg, Condition("1111"), 0, 0)
        ga._apply_mutation(cl, cfg, Perception("1111"))
        assert cl.is_more_general(Classifier(cfg, Condition("1111"), 0, 0))

    @pytest.mark.parametrize("cond1, cond2, x, y, end_cond1, end_cond2", [
        ("11111", "#####", 0, 5, "#####", "11111"),
        ("11111", "000", 0, 5, "00011", "111"),
        ("11111", "#####", 1, 4, "1###1", "#111#")
    ])
    def test_crossover_area(self, cfg, ga, cond1, cond2, x, y, end_cond1, end_cond2):
        cl1 = Classifier(cfg, Condition(cond1), 0, 0)
        cl2 = Classifier(cfg, Condition(cond2), 1, 0)
        ga._apply_crossover_in_area(cl1, cl2, x, y)
        assert cl1.condition == Condition(end_cond1)
        assert cl2.condition == Condition(end_cond2)
        # Just to check for errors

    def test_crossover_values(self, cfg, ga, situation, classifiers_list_diff_actions):
        cl1 = Classifier(cfg, Condition(situation), 0, 0)
        cl2 = Classifier(cfg, Condition(situation), 1, 0)
        ga._apply_crossover(
            cl1, cl2,
            classifiers_list_diff_actions[0],
            classifiers_list_diff_actions[1]
            )
        assert cl1.prediction == cl2.prediction
        assert cl1.error == cl2.error
        assert cl1.fitness == cl2.fitness

    @pytest.mark.parametrize("chi", [
        1,
        0
    ])
    # only tests for errors and types
    def test_run_ga(self, cfg, ga, classifiers_list_diff_actions, chi):
        cfg.do_GA_subsumption = True
        xcs = XCS(cfg, classifiers_list_diff_actions)
        action_set = xcs.population.generate_action_set(0)
        cfg.chi = chi  # do perform crossover
        cfg.do_GA_subsumption = False  # do not perform subsumption
        ga.run_ga(action_set, Perception("0000"), 100000)
        assert xcs.population[0].time_stamp != 0
        assert xcs.population.numerosity > 4

    def test_make_children(self, cfg, ga, classifiers_list_diff_actions):
        child1, child2 = ga._make_children(
            classifiers_list_diff_actions[0],
            classifiers_list_diff_actions[1],
            0
        )
        assert child1.numerosity == 1
        assert child2.numerosity == 1
        assert child1.experience == 0
        assert child2.experience == 0
        assert id(child1) != id(classifiers_list_diff_actions[0])
        assert id(child2) != id(classifiers_list_diff_actions[0])
        assert id(child1) != id(classifiers_list_diff_actions[1])
        assert id(child2) != id(classifiers_list_diff_actions[1])

    def test_do_ga_subsumption_does_subsume_true(self, cfg, ga, classifiers_list_diff_actions, situation):
        cfg.do_GA_subsumption = True
        classifiers_list_diff_actions[0].error = 0
        classifiers_list_diff_actions[1].error = 0
        classifiers_list_diff_actions[0].expirience = 30
        classifiers_list_diff_actions[1].expirience = 30
        ga._perform_insertion_or_subsumption(
            Classifier(cfg, Condition(situation), 0, 0),
            Classifier(cfg, Condition(situation), 1, 0),
            classifiers_list_diff_actions[0],
            classifiers_list_diff_actions[1]
        )
        assert classifiers_list_diff_actions[0].numerosity > 1
        assert classifiers_list_diff_actions[1].numerosity > 1

    def test_do_ga_subsumption_does_subsume_false(self, cfg, ga, classifiers_list_diff_actions, situation):
        cfg.do_GA_subsumption = True
        ga._perform_insertion_or_subsumption(
            Classifier(cfg, Condition("####"), 0, 0),
            Classifier(cfg, Condition("####"), 1, 0),
            classifiers_list_diff_actions[0],
            classifiers_list_diff_actions[1]
        )
        assert classifiers_list_diff_actions[0].numerosity == 1
        assert classifiers_list_diff_actions[1].numerosity == 1
        assert len(classifiers_list_diff_actions) == 6

