import pytest

from lcs import Perception
from lcs.agents.acs2 import Configuration, ClassifiersList, \
    Classifier


class TestClassifierList:

    @pytest.fixture
    def cfg(self):
        return Configuration(8, 8)

    def test_should_deny_insertion_illegal_types(self, cfg):
        population = ClassifiersList()

        with pytest.raises(TypeError):
            # Try to insert an integer instead of classifier object
            population.append(4)

    def test_should_insert_classifier(self, cfg):
        # given
        population = ClassifiersList()
        cl = Classifier(cfg=cfg)

        # when
        population.append(cl)

        # then
        assert len(population) == 1

    def test_should_form_match_set(self, cfg):
        # given
        cl_1 = Classifier(cfg=cfg)
        cl_2 = Classifier(condition='1###0###', cfg=cfg)
        cl_3 = Classifier(condition='0###1###', cfg=cfg)

        population = ClassifiersList(*[cl_1, cl_2, cl_3])
        p0 = Perception('11110000')

        # when
        match_set = ClassifiersList.form_match_set(population, p0)

        # then
        assert len(match_set) == 2
        assert cl_1 in match_set
        assert cl_2 in match_set

    def test_should_form_action_set(self, cfg):
        # given
        cl_1 = Classifier(action=0, cfg=cfg)
        cl_2 = Classifier(action=0, cfg=cfg)
        cl_3 = Classifier(action=1, cfg=cfg)

        population = ClassifiersList(*[cl_1, cl_2, cl_3])
        action = 0

        # when
        action_set = ClassifiersList.form_action_set(population, action)

        # then
        assert len(action_set) == 2
        assert cl_1 in action_set
        assert cl_2 in action_set

    def test_should_expand(self, cfg):
        # given
        cl_1 = Classifier(action=0, cfg=cfg)
        cl_2 = Classifier(action=1, numerosity=2, cfg=cfg)
        cl_3 = Classifier(action=2, numerosity=3, cfg=cfg)
        population = ClassifiersList(*[cl_1, cl_2, cl_3])

        # when
        expanded = population.expand()

        # then
        assert len(expanded) == 6
        assert cl_1 in expanded
        assert cl_2 in expanded
        assert cl_3 in expanded

    def test_should_calculate_maximum_fitness(self, cfg):
        # given
        population = ClassifiersList()

        # when & then
        # C1 - does not anticipate change
        c1 = Classifier(cfg=cfg)
        population.append(c1)
        assert 0.0 == population.get_maximum_fitness()

        # when & then
        # C2 - does anticipate some change
        c2 = Classifier(effect='1###0###',
                        reward=0.25,
                        cfg=cfg)
        population.append(c2)
        assert 0.125 == population.get_maximum_fitness()

        # when & then
        # C3 - does anticipate change and is quite good
        c3 = Classifier(effect='1#######',
                        quality=0.8,
                        reward=5,
                        cfg=cfg)
        population.append(c3)
        assert 4 == population.get_maximum_fitness()

    def test_should_apply_reinforcement_learning(self, cfg):
        # given
        cl = Classifier(reward=34.29, intermediate_reward=11.29, cfg=cfg)
        population = ClassifiersList(*[cl])

        # when
        ClassifiersList.apply_reinforcement_learning(
            population, 0, 28.79, cfg.beta, cfg.gamma)

        # then
        assert abs(33.94 - cl.r) < 0.1
        assert abs(10.74 - cl.ir) < 0.1

    def test_should_form_match_set_backwards(self, cfg):
        # given
        population = ClassifiersList()
        situation = Perception('11110000')

        # C1 - general condition
        c1 = Classifier(cfg=cfg)

        # C2 - matching
        c2 = Classifier(condition='0##0####', effect='1##1####', cfg=cfg)

        # C3 - non-matching
        c3 = Classifier(condition='0###1###', effect='1######0', cfg=cfg)

        # C4 - non-matching
        c4 = Classifier(condition='0###0###', effect='1###1###', cfg=cfg)

        population.append(c1)
        population.append(c2)
        population.append(c3)
        population.append(c4)

        # when
        match_set = ClassifiersList.form_match_set_backwards(population,
                                                             situation)
        # then
        assert 2 == len(match_set)
        assert c1 in match_set
        assert c2 in match_set

    def test_get_quality_classifiers_list(self, cfg):
        # given
        population = ClassifiersList()

        # C1 - matching
        c1 = Classifier(quality=0.9, cfg=cfg)

        # C2 - matching
        c2 = Classifier(quality=0.7, cfg=cfg)

        # C3 - non-matching
        c3 = Classifier(quality=0.5, cfg=cfg)

        # C4 - non-matching
        c4 = Classifier(quality=0.1, cfg=cfg)

        population.append(c1)
        population.append(c2)
        population.append(c3)
        population.append(c4)

        # when
        match_set = population.get_quality_classifiers_list(0.5, cfg)
        # then
        assert 2 == len(match_set)
        assert c1 in match_set
        assert c2 in match_set

    def test_exists_classifier(self, cfg):
        # given
        population = ClassifiersList()
        prev_situation = Perception('01100000')
        situation = Perception('11110000')
        act = 0
        q = 0.5

        # C1 - OK
        c1 = Classifier(condition='0##0####', action=0, effect='1##1####',
                        quality=0.7, cfg=cfg)

        # C2 - wrong action
        c2 = Classifier(condition='0##0####', action=1, effect='1##1####',
                        quality=0.7, cfg=cfg)

        # C3 - wrong condition
        c3 = Classifier(condition='0##1####', action=0, effect='1##1####',
                        quality=0.7, cfg=cfg)

        # C4 - wrong effect
        c4 = Classifier(condition='0##0####', action=0, effect='1##0####',
                        quality=0.7, cfg=cfg)

        # C5 - wrong quality
        c5 = Classifier(condition='0##0####', action=0, effect='1##1####',
                        quality=0.25, cfg=cfg)

        population.append(c2)
        population.append(c3)
        population.append(c4)
        population.append(c5)

        # when
        result0 = population.\
            exists_classifier(previous_situation=prev_situation,
                              situation=situation, action=act, quality=q)

        population.append(c1)
        result1 = population.\
            exists_classifier(previous_situation=prev_situation,
                              situation=situation, action=act, quality=q)

        # then
        assert result0 is False
        assert result1 is True
