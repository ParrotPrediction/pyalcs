import pytest

from lcs.agents.acs2 import Configuration, Classifier, \
    Condition
from lcs.agents.acs2.components.genetic_algorithm \
    import mutate, two_point_crossover
from tests.randommock import RandomMock, SampleMock


class TestGeneticAlgorithm:

    @pytest.fixture
    def cfg(self):
        return Configuration(8, 8)

    def test_mutate_1(self, cfg):
        # given
        cl = Classifier(Condition('##011###', cfg), cfg=cfg)
        s = cfg.mu * 0.5  # less then MU
        b = 1 - (1 - cfg.mu) * 0.5  # more then MU

        # when
        mutate(cl, cfg.mu, randomfunc=RandomMock([s, b, b]))

        # then
        assert Condition('###11###') == cl.condition

    def test_mutate_2(self, cfg):
        # given
        cl = Classifier(Condition('##011###'), cfg=cfg)
        s = cfg.mu * 0.5  # less then MU
        b = 1 - (1 - cfg.mu) * 0.5  # more then MU

        # when
        mutate(cl, cfg.mu, randomfunc=RandomMock([b, b, s]))

        # then
        assert Condition('##01####') == cl.condition

    def test_copy_from_and_mutate_does_not_influence_another_condition(self,
                                                                       cfg):
        """ Verify that not just reference to Condition copied (changing which
        will change the original - definitily not original C++ code did). """
        # given
        s = cfg.mu * 0.5  # less then MU
        b = 1 - (1 - cfg.mu) * 0.5  # more then MU

        operation_time = 123
        original_cl = Classifier(
            condition='1###1011',
            cfg=cfg
        )

        copied_cl = Classifier.copy_from(original_cl, operation_time)

        # when
        mutate(copied_cl, cfg.mu, RandomMock([s, b, b, b, b]))

        # then
        assert Condition('####1011') == copied_cl.condition
        assert Condition('1###1011') == original_cl.condition

        # when
        mutate(original_cl, cfg.mu, RandomMock([b, s, b, b, b]))

        # then
        assert Condition('1####011') == original_cl.condition
        assert Condition('####1011') == copied_cl.condition

    def test_crossover(self, cfg):
        # given
        cl1 = Classifier(condition='0##10###', cfg=cfg)
        cl2 = Classifier(condition='#10##0##', cfg=cfg)

        # when
        two_point_crossover(cl1, cl2, samplefunc=SampleMock([1, 4]))

        # then
        assert Condition('010#0###') == cl1.condition
        assert Condition('###1#0##') == cl2.condition

    def test_crossover_allows_to_change_last_element(self, cfg):
        # given
        cl1 = Classifier(condition='0##10###', cfg=cfg)
        cl2 = Classifier(condition='#10##011', cfg=cfg)

        # when
        two_point_crossover(cl1, cl2, samplefunc=SampleMock([5, 8]))

        # then
        assert Condition('0##10011') == cl1.condition
        assert Condition('#10#####') == cl2.condition
