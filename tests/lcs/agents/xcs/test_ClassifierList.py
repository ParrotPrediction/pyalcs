import pytest

from copy import copy
from lcs import Perception
from lcs.agents.xcs import Configuration, Condition, Classifier, ClassifiersList


class TestClassifiersList:

    @pytest.fixture
    def cfg(self):
        return Configuration(number_of_actions=4)

    @pytest.fixture
    def situation(self):
        return "1100"

    @pytest.fixture
    def classifiers_list_diff_actions(self, cfg, situation):
        pop = ClassifiersList(cfg)
        pop.insert_in_population(Classifier(cfg, Condition(situation), 0, 0))
        pop.insert_in_population(Classifier(cfg, Condition(situation), 1, 0))
        pop.insert_in_population(Classifier(cfg, Condition(situation), 2, 0))
        pop.insert_in_population(Classifier(cfg, Condition(situation), 3, 0))
        return pop

    def test_should_be_empty_when_initialized(self, cfg):
        assert len(ClassifiersList(cfg)) == 0

    @pytest.mark.parametrize("cond, act", [
        ("1100", 0),
        ("1100", 1),
        ("1100", 2),
        ("1100", 3)
    ])
    def test_insert_population(self, classifiers_list_diff_actions, cfg, cond, act):
        # given
        initial_size = len(classifiers_list_diff_actions)
        cl = Classifier(cfg=cfg, condition=Condition(cond), action=act, time_stamp=0)

        # when
        classifiers_list_diff_actions.insert_in_population(cl)

        # then
        assert len(classifiers_list_diff_actions) == initial_size
        assert any(c == cl for c in classifiers_list_diff_actions)
        assert any(c.numerosity == 2 for c in classifiers_list_diff_actions)

    @pytest.mark.parametrize("cond1, cond2, act1, act2, size", [
        ("1111", "1111", 1, 1, 5),
        ("#100", "#100", 0, 0, 5),
        ("0##11#", "0##11#", 0, 0, 5),

    ])
    def test_insert_population_two(self, cfg, classifiers_list_diff_actions, cond1, cond2, act1, act2, size):
        # given
        cl1 = Classifier(condition=Condition(cond1), action=act1, cfg=cfg)
        cl2 = Classifier(condition=Condition(cond2), action=act2, cfg=cfg)

        # when
        classifiers_list_diff_actions.insert_in_population(cl1)
        classifiers_list_diff_actions.insert_in_population(cl2)

        # then
        assert any(c == cl1 for c in classifiers_list_diff_actions)
        assert any(c == cl2 for c in classifiers_list_diff_actions)
        assert len(classifiers_list_diff_actions) == size

    @pytest.mark.parametrize("cond, act", [
        ("1111", 0),
        ("123", 1),
        ("A", 2),
        ("##11", 0),
        ("#11#", 0),
        ("112", 0),
    ])
    def test_insert_population_new_condition(self, classifiers_list_diff_actions, cfg, cond, act):
        # given
        cl = Classifier(condition=Condition("1111"), action=0, cfg=cfg)

        # when
        classifiers_list_diff_actions.insert_in_population(cl)

        # then
        for c in classifiers_list_diff_actions:
            assert c.numerosity == 1
        assert classifiers_list_diff_actions[4] == cl

    def test_covering(self, cfg):
        classifiers_list = ClassifiersList(cfg)
        covering_cl = classifiers_list.generate_covering_classifier(Perception("1111"), 0, 0)
        assert covering_cl.does_match(Perception("1111"))
        assert classifiers_list.generate_covering_classifier("1111", 0, 0).action == 0

    def test_deletion(self, cfg: Configuration):
        classifiers_list = ClassifiersList(cfg)
        for i in range(cfg.max_population + 1):
            classifiers_list.insert_in_population(
                Classifier(cfg, Condition("1111"), 0, 0)
            )
        assert sum(cl.numerosity for cl in classifiers_list) > cfg.max_population
        classifiers_list.delete_from_population()
        assert sum(cl.numerosity for cl in classifiers_list) <= cfg.max_population

    def test_removes_correct_one(self, cfg):
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1111"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1110"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("0000"), 0, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1000"), 3, 0))
        deletion_votes = []
        for _ in range(len(classifiers_list)):
            deletion_votes.append(1)
        selector = sum(deletion_votes) / 2
        classifiers_list._remove_based_on_votes(deletion_votes, selector)
        assert not any(cl.does_match("0000") for cl in classifiers_list)
        assert len(classifiers_list) == 4

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

    def test_find_not_present_action(self, cfg):
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 1, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 2, 0))
        classifiers_list.insert_in_population(Classifier(cfg, Condition("1100"), 3, 0))
        assert 0 == classifiers_list._find_not_present_action(classifiers_list)

    def test_prediction_array(self, cfg, classifiers_list_diff_actions):
        classifiers_list_diff_actions[0].prediction = 10
        prediction_array = classifiers_list_diff_actions.prediction_array
        assert len(classifiers_list_diff_actions) == cfg.number_of_actions
        assert prediction_array[0] > prediction_array[1]

    def test_update_fitness(self, cfg, classifiers_list_diff_actions):
        classifiers_list_diff_actions._update_fitness()
        for cl in classifiers_list_diff_actions:
            assert cl.fitness != cfg.initial_fitness

    def test_update_set(self, cfg: Configuration, classifiers_list_diff_actions):
        cfg.do_GA_subsumption = True
        cl = copy(classifiers_list_diff_actions[0])
        cfg.learning_rate = 1
        classifiers_list_diff_actions.update_set(0.2)
        for c in classifiers_list_diff_actions:
            assert c.experience > 0
            assert c.prediction != cl.prediction
            assert c.error != cl.error
