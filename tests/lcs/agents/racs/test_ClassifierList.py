import pytest

from lcs import Perception
from lcs.agents.racs import Configuration, Condition, \
    Classifier, ClassifiersList, Effect
from lcs.representations import Interval


class TestClassifiersList:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2)

    def test_should_initialize_classifier_list(self, cfg):
        # given
        cl1 = Classifier(action=1, cfg=cfg)
        cl2 = Classifier(action=2, cfg=cfg)
        cl3 = Classifier(action=3, cfg=cfg)

        # when
        cll = ClassifiersList(*[cl1, cl2])

        # then
        assert len(cll) == 2
        assert cl1 in cll
        assert cl2 in cll
        assert cl3 not in cll

    def test_should_form_match_set(self, cfg):
        # given
        observation = Perception([.2, .6], oktypes=(float,))

        cl1 = Classifier(
            condition=Condition([Interval(.1, .3), Interval(.5, .7)], cfg=cfg),
            cfg=cfg)
        cl2 = Classifier(
            condition=Condition([Interval(.1, .4), Interval(.2, .3)], cfg=cfg),
            cfg=cfg)
        cl3 = Classifier(cfg=cfg)

        population = ClassifiersList(*[cl1, cl2, cl3])

        # when
        match_set = population.form_match_set(observation)

        # then
        assert len(match_set) == 2
        assert cl1 in match_set
        assert cl2 not in match_set
        assert cl3 in match_set

    def test_should_form_action_set(self, cfg):
        # given
        cl1 = Classifier(action=0, cfg=cfg)
        cl2 = Classifier(action=0, cfg=cfg)
        cl3 = Classifier(action=1, cfg=cfg)

        population = ClassifiersList(*[cl1, cl2, cl3])

        # when
        action_set = population.form_action_set(0)

        # then
        assert len(action_set) == 2
        assert cl1 in action_set
        assert cl2 in action_set
        assert cl3 not in action_set

    def test_should_expand(self, cfg):
        # given
        cl1 = Classifier(numerosity=1, cfg=cfg)
        cl2 = Classifier(numerosity=2, cfg=cfg)
        cl3 = Classifier(numerosity=3, cfg=cfg)

        population = ClassifiersList(*[cl1, cl2, cl3])

        # when
        expanded = population.expand()

        # then
        assert len(expanded) == 6
        assert cl1 in expanded
        assert cl2 in expanded
        assert cl3 in expanded

    def test_should_get_maximum_fitness(self, cfg):
        # given
        # anticipate change - low fitness
        cl1 = Classifier(
            effect=Effect([Interval(0., 1.), Interval(0., .3)], cfg),
            quality=0.3, reward=1,
            cfg=cfg)

        # do not anticipate change - high fitness
        cl2 = Classifier(
            effect=Effect([Interval(0., 1.), Interval(0., 1.)], cfg),
            quality=0.5, reward=1,
            cfg=cfg)

        # anticipate change - medium fitness
        cl3 = Classifier(
            effect=Effect([Interval(0., .9), Interval(0., 1.)], cfg),
            quality=0.4, reward=1,
            cfg=cfg)

        population = ClassifiersList(*[cl1, cl2, cl3])

        # when
        mf = population.get_maximum_fitness()

        # then
        assert mf == cl3.fitness

    def test_should_return_zero_max_fitness(self, cfg):
        # given classifiers that does not anticipate change
        cl1 = Classifier(
            effect=Effect([Interval(0., 1.), Interval(0., 1.)], cfg),
            quality=0.5, reward=1,
            cfg=cfg)

        cl2 = Classifier(
            effect=Effect([Interval(0., 1.), Interval(0., 1.)], cfg),
            quality=0.7, reward=1,
            cfg=cfg)

        population = ClassifiersList(*[cl1, cl2])

        # when
        mf = population.get_maximum_fitness()

        # then
        assert mf == 0.0

    def test_should_apply_reinforcement_learning(self, cfg):
        # given
        cl = Classifier(
            reward=34.29,
            immediate_reward=11.29,
            cfg=cfg)

        population = ClassifiersList(*[cl])

        # when
        ClassifiersList.apply_reinforcement_learning(
            population, 0, 28.79, cfg.beta, cfg.gamma)

        # then
        assert abs(33.94 - cl.r) < 0.1
        assert abs(10.74 - cl.ir) < 0.1
