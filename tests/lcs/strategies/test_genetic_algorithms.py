import itertools
from dataclasses import dataclass

import pytest

import lcs.agents.acs2 as acs2
import lcs.strategies.genetic_algorithms as ga


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

    def test_two_point_crossover(self):
        # given
        cfg = acs2.Configuration(
            classifier_length=4, number_of_possible_actions=2)
        cl1 = acs2.Classifier(condition='1111', cfg=cfg)
        cl2 = acs2.Classifier(condition='2222', cfg=cfg)

        # when
        ga.two_point_crossover(cl1, cl2)

        # then
        assert '1' in cl1.condition
        assert '2' in cl1.condition
        assert '1' in cl2.condition
        assert '2' in cl2.condition

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
        population = acs2.ClassifiersList(*clss, cfg=cfg)
        action_set = population.form_action_set(1, cfg)

        ga.delete_classifiers(population, None, action_set, insize, theta_as)

        # then
        assert sum(cl.num for cl in population) == 18
        assert sum(cl.num for cl in action_set) == 8

    def test_should_not_find_old_classifier(self):
        pass

    def test_should_find_subsumer(self):
        # Among nonsubsumers
        pass

    def test_should_find_most_general_subsumer(self):
        pass

    def test_should_find_similar_classifier(self):
        pass

    def test_should_find_similar_among_subsumer(self):
        # When there are both subsumers and similar classifiers
        pass
