import pytest

from lcs.acs2 import ACS2Configuration, Classifier, ClassifiersList, Condition
from lcs.components.genetic_algorithm \
    import roulette_wheel_parents_selection, mutate, two_point_crossover
from tests.randommock import RandomMock, SampleMock


class TestGeneticAlgorithm:

    @pytest.fixture
    def cfg(self):
        return ACS2Configuration(8, 8)

    def test_should_select_parents1(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)
        c0 = Classifier(condition='######00', cfg=cfg)
        c1 = Classifier(condition='######01', cfg=cfg)
        c2 = Classifier(condition='######10', cfg=cfg)
        c3 = Classifier(condition='######11', cfg=cfg)
        population.append(c0)
        population.append(c1)
        population.append(c2)
        population.append(c3)

        # when
        p1, p2 = roulette_wheel_parents_selection(
            population, randomfunc=(RandomMock([0.7, 0.1])))

        # then
        assert c0 == p1

        # when
        p1, p2 = roulette_wheel_parents_selection(
            population, randomfunc=(RandomMock([0.3, 0.6])))

        # then
        assert c1 == p1
        assert c2 == p2

        # when
        p1, p2 = roulette_wheel_parents_selection(
            population, randomfunc=(RandomMock([0.2, 0.8])))

        # then
        assert c0 == p1
        assert c3 == p2

    def test_quality_and_numerosity_influence_parent_selection(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)
        c0 = Classifier(condition='######00',
                        quality=1,
                        numerosity=1,
                        cfg=cfg)
        c1 = Classifier(condition='######01', cfg=cfg)
        c2 = Classifier(condition='######10', cfg=cfg)
        population.append(c0)  # q3num = 1
        population.append(c1)  # q3num = 0.0625
        population.append(c2)  # q3num = 0.0625

        # when
        p1, p2 = roulette_wheel_parents_selection(
            population, randomfunc=(RandomMock([0.888, 0.999])))

        # then
        assert c1 == p1
        assert c2 == p2

        # when
        p1, p2 = roulette_wheel_parents_selection(
            population, randomfunc=(RandomMock([0.888, 0.777])))

        # then
        assert c0 == p1
        assert c1 == p2

    def test_mutate_1(self, cfg):
        # given
        cl = Classifier(Condition('##011###', cfg), cfg=cfg)
        s = cfg.mu * 0.5  # less then MU
        b = 1 - (1 - cfg.mu) * 0.5  # more then MU

        # when
        mutate(cl, cfg.mu, randomfunc=RandomMock([s, b, b]))

        # then
        assert Condition('###11###', cfg) == cl.condition

    def test_mutate_2(self, cfg):
        # given
        cl = Classifier(Condition('##011###', cfg), cfg=cfg)
        s = cfg.mu * 0.5  # less then MU
        b = 1 - (1 - cfg.mu) * 0.5  # more then MU

        # when
        mutate(cl, cfg.mu, randomfunc=RandomMock([b, b, s]))

        # then
        assert Condition('##01####', cfg) == cl.condition

    def test_copy_from_and_mutate_does_not_influence_another_condition(self,
                                                                       cfg):
        """ Verify that not just reference to Condition copied (changing which
        will change the original - definitily not original C++ code did). """
        # given
        s = cfg.mu * 0.5  # less then MU
        b = 1 - (1 - cfg.mu) * 0.5  # more then MU

        operation_time = 123
        original_cl = Classifier(
            condition=Condition('1###1011', cfg),
            cfg=cfg
        )

        copied_cl = Classifier.copy_from(original_cl, operation_time)

        # when
        mutate(copied_cl, cfg.mu, RandomMock([s, b, b, b, b]))

        # then
        assert Condition('####1011', cfg) == copied_cl.condition
        assert Condition('1###1011', cfg) == original_cl.condition

        # when
        mutate(original_cl, cfg.mu, RandomMock([b, s, b, b, b]))

        # then
        assert Condition('1####011', cfg) == original_cl.condition
        assert Condition('####1011', cfg) == copied_cl.condition

    def test_crossover(self, cfg):
        # given
        cl1 = Classifier(Condition('0##10###', cfg), cfg=cfg)
        cl2 = Classifier(Condition('#10##0##', cfg), cfg=cfg)

        # when
        two_point_crossover(cl1, cl2, samplefunc=SampleMock([1, 4]))

        # then
        assert Condition('010#0###', cfg) == cl1.condition
        assert Condition('###1#0##', cfg) == cl2.condition

    def test_crossover_allows_to_change_last_element(self, cfg):
        # given
        cl1 = Classifier(Condition('0##10###', cfg), cfg=cfg)
        cl2 = Classifier(Condition('#10##011', cfg), cfg=cfg)

        # when
        two_point_crossover(cl1, cl2, samplefunc=SampleMock([5, 8]))

        # then
        assert Condition('0##10011', cfg) == cl1.condition
        assert Condition('#10#####', cfg) == cl2.condition
