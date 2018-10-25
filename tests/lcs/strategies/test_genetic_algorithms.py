import itertools
from dataclasses import dataclass

import numpy as np
import pytest

import lcs.agents.acs2 as acs2
import lcs.strategies.genetic_algorithms as ga
from lcs import Perception


@dataclass(unsafe_hash=True)
class IdClassifier:
    id: int
    q: float


@dataclass(unsafe_hash=True)
class SimpleClassifier:
    q: float
    tav: float = 0.0
    marked: bool = False

    def is_marked(self):
        return self.marked


class TestGeneticAlgorithms:

    def test_roulette_wheel_selection(self):
        # given
        cl1 = IdClassifier(1, 0.7)
        cl2 = IdClassifier(2, 0.3)
        cl3 = IdClassifier(3, 0.1)
        pop = [cl1, cl2, cl3]

        def fitnessfcn(cl):
            return pow(cl.q, 3)

        # when
        n = 1000
        results = []
        for _ in range(n):
            results.extend(ga.roulette_wheel_selection(pop, fitnessfcn))

        # then
        results = sorted(results, key=lambda el: el.id)
        stats = {k: len(list(g)) for k, g in itertools.groupby(
            results, key=lambda el: el.id)}

        assert stats[cl1.id] > stats[cl2.id] * 10 > stats[cl3.id] * 10

    @pytest.mark.parametrize("_mu, _cond1, _cond2", [
        (0.0, '1234', '1234'),
        (1.0, '1234', '####')
    ])
    def test_generalizing_mutation(self, _mu, _cond1, _cond2):
        # given
        cfg = acs2.Configuration(
            classifier_length=4, number_of_possible_actions=2)
        cl = acs2.Classifier(condition=_cond1, cfg=cfg)

        # when
        ga.generalizing_mutation(cl, mu=_mu)

        # then
        assert cl.condition == acs2.Condition(_cond2)

    @pytest.mark.parametrize("_seed, _c1, _c2, _rc1, _rc2", [
        (111, '1111', '2222', '1211', '2122'),  # left=1, right=2
        (335, '1111', '2222', '2211', '1122'),  # left=0, right=2
        (575, '1111', '2222', '2222', '1111'),  # left=0, right=4
    ])
    def test_two_point_crossover(self, _seed, _c1, _c2, _rc1, _rc2):
        # given
        cfg = acs2.Configuration(
            classifier_length=4, number_of_possible_actions=2)
        cl1 = acs2.Classifier(condition=_c1, cfg=cfg)
        cl2 = acs2.Classifier(condition=_c2, cfg=cfg)

        # when
        np.random.seed(_seed)
        ga.two_point_crossover(cl1, cl2)

        # then
        assert cl1.condition == acs2.Condition(_rc1)
        assert cl2.condition == acs2.Condition(_rc2)

    @pytest.mark.parametrize(
        "_cl_del_q, _cl_del_marked, _cl_del_tav," +
        "_cl_q, _cl_marked, _cl_tav," +
        "_cl_selected", [
            # huge difference in `q`
            (.3, False, 0.0, .5, False, 0.0, False),
            (.3, False, 0.0, .1, False, 0.0, True),
            # cl is much better
            (.3, False, 0.0, .6, False, 0.0, False),
            # similar, cl marked
            (.45, False, 0.0, .5, True, 0.0, True),
            # similar, cl not marked, cl_del marked => tav dependent
            (.45, True, 0.0, .5, False, 1.0, True),
            (.45, True, 2.0, .5, False, 1.0, False),
            # similar, both unmarked, => tav dependent
            (.45, False, 1.0, .5, False, 1.1, True),
            (.45, False, 1.0, .5, False, 0.9, False),
            # similar, both marked. => tav dependent
            (.45, True, 0.0, 0.5, True, 1.0, True),
            (.45, True, 1.0, 0.5, True, 0.5, False),
            # similar, only cl_del marked => tav dependent
            (.45, True, 1.0, 0.5, False, 5.0, True),
            (.45, True, 5.0, 0.5, False, 1.0, False),
        ])
    def test_should_select_preferred_to_delete(self,
                                               _cl_q, _cl_marked, _cl_tav,
                                               _cl_del_q, _cl_del_marked,
                                               _cl_del_tav, _cl_selected):

        # given
        cl_del = SimpleClassifier(_cl_del_q, _cl_del_tav, _cl_del_marked)
        cl = SimpleClassifier(_cl_q, _cl_tav, _cl_marked)

        # then
        assert ga._is_preferred_to_delete(cl_del, cl) is _cl_selected

    def test_should_delete_ga_classifier(self):
        # given
        cfg = acs2.Configuration(
            classifier_length=4, number_of_possible_actions=2)
        clss = []

        # Total of 20 micro-classifiers
        action_cl_num_fixtures = {
            1: [1, 5, 4],
            2: [1, 1, 6, 2]
        }

        for action, num_lst in action_cl_num_fixtures.items():
            for num in num_lst:
                clss.append(acs2.Classifier(
                    action=action, numerosity=num, cfg=cfg))

        insize = 2  # space needed for new classifiers
        theta_as = 10  # maximum size of the action set

        # when
        population = acs2.ClassifiersList(*clss)
        action_set = population.form_action_set(1)

        ga.delete_classifiers(population, None, action_set, insize, theta_as)

        # then
        assert sum(cl.num for cl in population) == 18
        assert sum(cl.num for cl in action_set) == 8

    def test_should_not_find_old_classifier(self):
        # given
        cfg = acs2.Configuration(
            classifier_length=4, number_of_possible_actions=2)
        cl = acs2.Classifier(cfg=cfg)
        population = acs2.ClassifiersList()

        # when
        old_cl = ga._find_old_classifier(population, cl, True, cfg.theta_exp)

        # then
        assert old_cl is None

    def test_should_find_old_classifier_only_subsumer(self):
        # given
        cfg = acs2.Configuration(
            classifier_length=8, number_of_possible_actions=4)

        subsumer1 = acs2.Classifier(
            condition='1##0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        subsumer2 = acs2.Classifier(
            condition='#1#0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        most_general = acs2.Classifier(
            condition='###0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        nonsubsumer = acs2.Classifier(cfg=cfg)

        cl = acs2.Classifier(
            condition='11#0####',
            action=3,
            effect='##1#####',
            quality=0.5,
            reward=0.35,
            experience=1,
            cfg=cfg)

        population = acs2.ClassifiersList(
            *[nonsubsumer, subsumer1, nonsubsumer, most_general,
              subsumer2, nonsubsumer])

        # when
        old_cl = ga._find_old_classifier(population, cl, True, cfg.theta_exp)

        # then
        assert old_cl == most_general

    def test_find_old_classifier_only_similar(self):
        # given
        cfg = acs2.Configuration(
            classifier_length=4, number_of_possible_actions=2)

        cl_1 = acs2.Classifier(action=1, experience=32, cfg=cfg)
        cl_2 = acs2.Classifier(action=1, cfg=cfg)
        population = acs2.ClassifiersList(
            *[cl_1,
              acs2.Classifier(action=2, cfg=cfg),
              acs2.Classifier(action=3, cfg=cfg),
              cl_2])

        # when
        cl = acs2.Classifier(action=1, cfg=cfg)
        old_cl = ga._find_old_classifier(population, cl, True, cfg.theta_exp)

        # then
        assert old_cl == cl_1

    def test_find_old_classifier_similar_and_subsumer_subsumer_returned(self):
        # given
        cfg = acs2.Configuration(
            classifier_length=8, number_of_possible_actions=4)

        subsumer = acs2.Classifier(
            condition='1#######',
            action=1,
            experience=21,
            quality=0.95,
            cfg=cfg)

        similar = acs2.Classifier(
            condition='10######',
            action=1,
            cfg=cfg)

        population = acs2.ClassifiersList(*[similar, subsumer])

        cl = acs2.Classifier(
            condition='10######',
            action=1,
            cfg=cfg)

        # when
        old_cls = ga._find_old_classifier(population, cl, True, cfg.theta_exp)

        # then
        assert similar == cl
        assert subsumer == old_cls

    def test_should_add_classifier(self):
        # given
        cfg = acs2.Configuration(
            classifier_length=8, number_of_possible_actions=4)

        cl_1 = acs2.Classifier(action=1, cfg=cfg)
        cl_3 = acs2.Classifier(action=3, cfg=cfg)
        cl_4 = acs2.Classifier(action=4, cfg=cfg)

        population = acs2.ClassifiersList(*[cl_1, cl_3, cl_4])
        match_set = acs2.ClassifiersList()
        action_set = acs2.ClassifiersList(*[cl_1])

        p0 = Perception('10101010')
        cl = acs2.Classifier(
            action=1,
            condition='1#######',
            cfg=cfg)

        # when
        ga.add_classifier(cl, p0,
                          population, match_set, action_set,
                          do_subsumption=True, theta_exp=cfg.theta_exp)

        # then
        assert acs2.ClassifiersList(*[cl_1, cl_3, cl_4, cl]) == population
        assert acs2.ClassifiersList(*[cl]) == match_set
        assert acs2.ClassifiersList(*[cl_1, cl]) == action_set

    def test_add_ga_classifier_increase_numerosity(self):
        # given
        cfg = acs2.Configuration(
            classifier_length=8, number_of_possible_actions=4)

        cl_1 = acs2.Classifier(action=2, condition='1#######', cfg=cfg)
        cl_2 = acs2.Classifier(action=3, cfg=cfg)
        cl_3 = acs2.Classifier(action=4, cfg=cfg)

        population = acs2.ClassifiersList(*[cl_1, cl_2, cl_3])
        match_set = acs2.ClassifiersList(*[cl_1])
        action_set = acs2.ClassifiersList(*[cl_1])

        cl = acs2.Classifier(action=2, condition='1#######', cfg=cfg)

        # when
        p0 = Perception('10101010')
        ga.add_classifier(cl, p0,
                          population, match_set, action_set,
                          do_subsumption=True, theta_exp=cfg.theta_exp)

        # then
        assert cl_1.num == 2
        assert acs2.ClassifiersList(*[cl_1, cl_2, cl_3]) == population
        assert acs2.ClassifiersList(*[cl_1]) == match_set
        assert acs2.ClassifiersList(*[cl_1]) == action_set
