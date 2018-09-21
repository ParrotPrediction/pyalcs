import pytest

from lcs import Perception
from lcs.strategies.subsumption import does_subsume
from lcs.agents.acs2 import Configuration, ClassifiersList, \
    Condition, Classifier


class TestClassifierList:

    @pytest.fixture
    def cfg(self):
        return Configuration(8, 8)

    @pytest.mark.skip(reason="move to another file")
    def test_find_subsumer_random_select_one_of_equally_general_sbsmers(self,
                                                                        cfg):
        # given
        subsumer1 = Classifier(condition='1##0####',
                               action=3,
                               effect='##1#####',
                               quality=0.93,
                               reward=1.35,
                               experience=23,
                               cfg=cfg)

        subsumer2 = Classifier(condition='#1#0####',
                               action=3,
                               effect='##1#####',
                               quality=0.93,
                               reward=1.35,
                               experience=23,
                               cfg=cfg)

        nonsubsumer = Classifier(cfg=cfg)

        classifier = Classifier(condition='11#0####',
                                action=3,
                                effect='##1#####',
                                quality=0.5,
                                reward=0.35,
                                experience=1,
                                cfg=cfg)

        classifiers_list = ClassifiersList(
            *[nonsubsumer, subsumer1, subsumer2, nonsubsumer])

        # when
        actual_subsumer = classifiers_list.find_subsumer(
            classifier, choice_func=lambda l: l[0])

        # then
        assert actual_subsumer == subsumer1

        # when
        actual_subsumer = classifiers_list.find_subsumer(
            classifier, choice_func=lambda l: l[1])

        # then
        assert actual_subsumer == subsumer2

    @pytest.mark.skip(reason="move to another file")
    def test_find_subsumer_selects_most_general_subsumer(self, cfg):
        # given
        subsumer1 = Classifier(
            condition='1##0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        subsumer2 = Classifier(
            condition='#1#0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35, experience=23,
            cfg=cfg)

        most_general = Classifier(
            condition='###0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        nonsubsumer = Classifier(cfg=cfg)

        classifier = Classifier(
            condition='11#0####',
            action=3,
            effect='##1#####',
            quality=0.5,
            reward=0.35,
            experience=1,
            cfg=cfg)

        classifiers_list = ClassifiersList(
            *[nonsubsumer, subsumer1, nonsubsumer, most_general,
              subsumer2, nonsubsumer])

        # when
        actual_subsumer = classifiers_list.find_subsumer(
            classifier, choice_func=lambda l: l[0])

        # then
        assert actual_subsumer == most_general

    @pytest.mark.skip(reason="move to another file")
    def test_find_old_classifier_only_subsumer(self, cfg):
        # given
        subsumer1 = Classifier(
            condition='1##0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        subsumer2 = Classifier(
            condition='#1#0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        most_general = Classifier(
            condition='###0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        nonsubsumer = Classifier(cfg=cfg)

        classifier = Classifier(
            condition='11#0####',
            action=3,
            effect='##1#####',
            quality=0.5,
            reward=0.35,
            experience=1,
            cfg=cfg)

        classifiers_list = ClassifiersList(
            *[nonsubsumer, subsumer1, nonsubsumer, most_general,
              subsumer2, nonsubsumer])

        # when
        actual_old_classifier = classifiers_list.find_old_classifier(
            classifier)

        # then
        assert most_general == actual_old_classifier

    @pytest.mark.skip(reason="move to another file")
    def test_find_old_classifier_only_similar(self, cfg):
        # given
        classifier_1 = Classifier(action=1, experience=32, cfg=cfg)
        classifier_2 = Classifier(action=1, cfg=cfg)
        classifiers = ClassifiersList(
            *[classifier_1,
              Classifier(action=2, cfg=cfg),
              Classifier(action=3, cfg=cfg),
              classifier_2])

        # when
        actual_old_classifier = classifiers.find_old_classifier(
            Classifier(action=1, cfg=cfg))

        # then
        assert classifier_1 == actual_old_classifier

    @pytest.mark.skip(reason="move to another file")
    def test_find_old_classifier_similar_and_subsumer_subsumer_returned(self,
                                                                        cfg):
        # given
        subsumer = Classifier(condition='1#######',
                              action=1,
                              experience=21,
                              quality=0.95,
                              cfg=cfg)

        similar = Classifier(condition='10######',
                             action=1,
                             cfg=cfg)

        existing_classifiers = ClassifiersList(*[similar, subsumer])

        classifier = Classifier(condition='10######',
                                action=1,
                                cfg=cfg)

        # when
        old_cls = existing_classifiers.find_old_classifier(classifier)

        # then
        assert does_subsume(subsumer, classifier, cfg.theta_exp) is True
        assert similar == classifier
        assert subsumer == old_cls

    @pytest.mark.skip(reason="move to another file")
    def test_find_old_classifier_none(self, cfg):
        # given
        classifier_list = ClassifiersList()
        cl = Classifier(cfg=cfg)

        # when
        old_cl = classifier_list.find_old_classifier(cl)

        assert old_cl is None

    @pytest.mark.skip(reason="move to another file")
    def test_add_ga_classifier_add(self, cfg):
        # given
        cl_1 = Classifier(action=1, cfg=cfg)
        cl_2 = Classifier(action=2,
                          condition='1#######',
                          cfg=cfg)
        cl_3 = Classifier(action=3, cfg=cfg)
        cl_4 = Classifier(action=4, cfg=cfg)
        action_set = ClassifiersList(*[cl_1])
        match_set = ClassifiersList()
        population = ClassifiersList(*[cl_1, cl_3, cl_4])

        # when
        action_set.add_ga_classifier(cl_2, match_set, population)

        # then
        assert ClassifiersList(*[cl_2]) == match_set
        assert ClassifiersList(*[cl_1, cl_3, cl_4, cl_2]) == population

    @pytest.mark.skip(reason="move to another file")
    def test_add_ga_classifier_increase_numerosity(self, cfg):
        # given
        cl_1 = Classifier(action=2,
                          condition='1#######',
                          cfg=cfg)
        cl_2 = Classifier(action=2,
                          condition='1#######',
                          cfg=cfg)
        cl_3 = Classifier(action=3, cfg=cfg)
        cl_4 = Classifier(action=4, cfg=cfg)

        action_set = ClassifiersList(*[cl_1])
        match_set = ClassifiersList(*[cl_1])
        population = ClassifiersList(*[cl_1, cl_3, cl_4])

        # when
        action_set.add_ga_classifier(cl_2, match_set, population)
        new_classifier = Classifier(
            action=2,
            condition=Condition('1#######'),
            numerosity=2,
            cfg=cfg)

        # then
        assert ClassifiersList(*[new_classifier, cl_3, cl_4]) == population

    def test_should_insert_classifier_1(self, cfg):
        population = ClassifiersList()

        with pytest.raises(TypeError):
            # Try to insert an integer instead of classifier object
            population.append(4)

    def test_should_insert_classifier_2(self, cfg):
        # given
        population = ClassifiersList()

        # when
        population.append(Classifier(cfg=cfg))

        # then
        assert 1 == len(population)

    def test_should_form_match_set(self, cfg):
        # given
        population = ClassifiersList()
        situation = Perception('11110000')

        # C1 - general condition
        c1 = Classifier(cfg=cfg)

        # C2 - matching condition
        c2 = Classifier(condition='1###0###', cfg=cfg)

        # C3 - non-matching condition
        c3 = Classifier(condition='0###1###', cfg=cfg)

        population.append(c1)
        population.append(c2)
        population.append(c3)

        # when
        match_set = ClassifiersList.form_match_set(population, situation)
        # then
        assert 2 == len(match_set)
        assert c1 in match_set
        assert c2 in match_set

    def test_should_form_action_set(self, cfg):
        # given
        population = ClassifiersList()
        c0 = Classifier(action=0, cfg=cfg)
        c01 = Classifier(action=0, cfg=cfg)
        c1 = Classifier(action=1, cfg=cfg)

        population.append(c0)
        population.append(c01)
        population.append(c1)

        # when & then
        action_set = ClassifiersList.form_action_set(population, 0)
        assert 2 == len(action_set)
        assert c0 in action_set
        assert c01 in action_set

        # when & then
        action_set = ClassifiersList.form_action_set(population, 1)
        assert 1 == len(action_set)
        assert c1 in action_set

    def test_should_expand(self, cfg):
        # given
        population = ClassifiersList()
        c0 = Classifier(action=0, cfg=cfg)
        c1 = Classifier(action=1, numerosity=2, cfg=cfg)
        c2 = Classifier(action=2, numerosity=3, cfg=cfg)

        population.append(c0)
        population.append(c1)
        population.append(c2)

        # when
        expanded = population.expand()

        # then
        assert len(expanded) == 6
        assert c0 in expanded
        assert c1 in expanded
        assert c2 in expanded

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
        population = ClassifiersList()
        c1 = Classifier(cfg=cfg)
        c1.r = 34.29
        c1.ir = 11.29
        population.append(c1)

        # when
        ClassifiersList.apply_reinforcement_learning(population,
                                                     0, 28.79,
                                                     cfg.beta, cfg.gamma)

        # then
        assert abs(33.94 - population[0].r) < 0.1
        assert abs(10.74 - population[0].ir) < 0.1
