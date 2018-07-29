import pytest

from lcs import Perception
from lcs.agents.acs2 import ACS2Configuration, ClassifiersList, \
    Condition, ACS2Classifier
from tests.randommock import RandomMock, SampleMock


class TestClassifierList:

    @pytest.fixture
    def cfg(self):
        return ACS2Configuration(8, 8)

    def test_find_subsumer_finds_single_subsumer(self, cfg):
        # given
        subsumer = ACS2Classifier(condition='###0####',
                                  action=3,
                                  effect='##1#####',
                                  quality=0.93,
                                  reward=1.35,
                                  experience=23,
                                  cfg=cfg)

        nonsubsumer = ACS2Classifier(cfg=cfg)

        classifiers_list = ClassifiersList(
            [nonsubsumer, subsumer, nonsubsumer], cfg)

        classifier = ACS2Classifier(condition='1##0####',
                                    action=3,
                                    effect='##1#####',
                                    quality=0.5,
                                    reward=0.35,
                                    experience=1,
                                    cfg=cfg)

        # when
        actual_subsumer = classifiers_list.find_subsumer(
            classifier, choice_func=lambda l: l[0])

        # then
        assert subsumer == actual_subsumer

    def test_find_subsumer_finds_single_subsumer_among_nonsubsumers(self, cfg):
        # given
        subsumer = ACS2Classifier(condition='###0####',
                                  action=3,
                                  effect='##1#####',
                                  quality=0.93,
                                  reward=1.35,
                                  experience=23,
                                  cfg=cfg)

        nonsubsumer = ACS2Classifier(cfg=cfg)

        classifier = ACS2Classifier(condition='1##0####',
                                    action=3,
                                    effect='##1#####',
                                    quality=0.5,
                                    reward=0.35,
                                    experience=1,
                                    cfg=cfg)
        classifiers_list = ClassifiersList(
            [nonsubsumer, subsumer, nonsubsumer], cfg)

        # when
        actual_subsumer = classifiers_list.find_subsumer(
            classifier, choice_func=lambda l: l[0])

        # then
        assert actual_subsumer == subsumer

    def test_find_subsumer_finds_selects_more_general_subsumer1(self, cfg):
        # given
        subsumer1 = ACS2Classifier(condition='1##0####',
                                   action=3,
                                   effect='##1#####',
                                   quality=0.93,
                                   reward=1.35,
                                   experience=23,
                                   cfg=cfg)
        subsumer2 = ACS2Classifier(condition='###0####',
                                   action=3,
                                   effect='##1#####',
                                   quality=0.93,
                                   reward=1.35,
                                   experience=23,
                                   cfg=cfg)

        nonsubsumer = ACS2Classifier(cfg=cfg)

        classifier = ACS2Classifier(condition='11#0####',
                                    action=3,
                                    effect='##1#####',
                                    quality=0.5,
                                    reward=0.35,
                                    experience=1,
                                    cfg=cfg)

        classifiers_list = ClassifiersList(
            [nonsubsumer, subsumer2, subsumer1, nonsubsumer], cfg)

        # when
        actual_subsumer = classifiers_list.find_subsumer(
            classifier, choice_func=lambda l: l[0])

        # then
        assert actual_subsumer == subsumer2

    def test_find_subsumer_finds_selects_more_general_subsumer2(self, cfg):
        # given
        subsumer1 = ACS2Classifier(
            condition='1##0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)
        subsumer2 = ACS2Classifier(
            condition='###0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        nonsubsumer = ACS2Classifier(cfg=cfg)

        classifier = ACS2Classifier(
            condition='11#0####',
            action=3,
            effect='##1#####',
            quality=0.5,
            reward=0.35,
            experience=1,
            cfg=cfg)

        classifiers_list = ClassifiersList(
            [nonsubsumer, subsumer1, subsumer2, nonsubsumer], cfg)

        # when
        actual_subsumer = classifiers_list.find_subsumer(
            classifier, choice_func=lambda l: l[0])

        # then
        assert actual_subsumer == subsumer2

    def test_find_subsumer_random_select_one_of_equally_general_sbsmers(self,
                                                                        cfg):
        # given
        subsumer1 = ACS2Classifier(condition='1##0####',
                                   action=3,
                                   effect='##1#####',
                                   quality=0.93,
                                   reward=1.35,
                                   experience=23,
                                   cfg=cfg)

        subsumer2 = ACS2Classifier(condition='#1#0####',
                                   action=3,
                                   effect='##1#####',
                                   quality=0.93,
                                   reward=1.35,
                                   experience=23,
                                   cfg=cfg)

        nonsubsumer = ACS2Classifier(cfg=cfg)

        classifier = ACS2Classifier(condition='11#0####',
                                    action=3,
                                    effect='##1#####',
                                    quality=0.5,
                                    reward=0.35,
                                    experience=1,
                                    cfg=cfg)

        classifiers_list = ClassifiersList(
            [nonsubsumer, subsumer1, subsumer2, nonsubsumer], cfg)

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

    def test_find_subsumer_selects_most_general_subsumer(self, cfg):
        # given
        subsumer1 = ACS2Classifier(
            condition='1##0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        subsumer2 = ACS2Classifier(
            condition='#1#0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35, experience=23,
            cfg=cfg)

        most_general = ACS2Classifier(
            condition='###0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        nonsubsumer = ACS2Classifier(cfg=cfg)

        classifier = ACS2Classifier(
            condition='11#0####',
            action=3,
            effect='##1#####',
            quality=0.5,
            reward=0.35,
            experience=1,
            cfg=cfg)

        classifiers_list = ClassifiersList(
            [nonsubsumer, subsumer1, nonsubsumer, most_general,
             subsumer2, nonsubsumer], cfg)

        # when
        actual_subsumer = classifiers_list.find_subsumer(
            classifier, choice_func=lambda l: l[0])

        # then
        assert actual_subsumer == most_general

    def test_find_old_classifier_only_subsumer(self, cfg):
        # given
        subsumer1 = ACS2Classifier(
            condition='1##0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        subsumer2 = ACS2Classifier(
            condition='#1#0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        most_general = ACS2Classifier(
            condition='###0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=cfg)

        nonsubsumer = ACS2Classifier(cfg=cfg)

        classifier = ACS2Classifier(
            condition='11#0####',
            action=3,
            effect='##1#####',
            quality=0.5,
            reward=0.35,
            experience=1,
            cfg=cfg)

        classifiers_list = ClassifiersList(
            [nonsubsumer, subsumer1, nonsubsumer, most_general,
             subsumer2, nonsubsumer], cfg)

        # when
        actual_old_classifier = classifiers_list.find_old_classifier(
            classifier)

        # then
        assert most_general == actual_old_classifier

    def test_find_old_classifier_only_similar(self, cfg):
        # given
        classifier_1 = ACS2Classifier(action=1, experience=32, cfg=cfg)
        classifier_2 = ACS2Classifier(action=1, cfg=cfg)
        classifiers = ClassifiersList(
            [classifier_1,
             ACS2Classifier(action=2, cfg=cfg),
             ACS2Classifier(action=3, cfg=cfg),
             classifier_2], cfg)

        # when
        actual_old_classifier = classifiers.find_old_classifier(
            ACS2Classifier(action=1, cfg=cfg))

        # then
        assert classifier_1 == actual_old_classifier

    def test_find_old_classifier_similar_and_subsumer_subsumer_returned(self,
                                                                        cfg):
        # given
        subsumer = ACS2Classifier(condition='1#######',
                                  action=1,
                                  experience=21,
                                  quality=0.95,
                                  cfg=cfg)

        similar = ACS2Classifier(condition='10######',
                                 action=1,
                                 cfg=cfg)

        existing_classifiers = ClassifiersList([similar, subsumer], cfg)

        classifier = ACS2Classifier(condition='10######',
                                    action=1,
                                    cfg=cfg)

        # when
        old_cls = existing_classifiers.find_old_classifier(classifier)

        # then
        assert subsumer.does_subsume(classifier) is True
        assert similar.is_similar(classifier)
        assert subsumer == old_cls

    def test_find_old_classifier_none(self, cfg):
        # given
        classifier_list = ClassifiersList([], cfg=cfg)
        cl = ACS2Classifier(cfg=cfg)

        # when
        old_cl = classifier_list.find_old_classifier(cl)

        assert old_cl is None

    def test_select_classifier_to_delete(self, cfg):
        # given
        selected_first = ACS2Classifier(quality=0.5, cfg=cfg)
        much_worse = ACS2Classifier(quality=0.2, cfg=cfg)
        yet_another_to_consider = ACS2Classifier(quality=0.2, cfg=cfg)
        classifiers = ClassifiersList(
            [ACS2Classifier(cfg=cfg),
             selected_first,
             ACS2Classifier(cfg=cfg),
             much_worse,
             yet_another_to_consider,
             ACS2Classifier(cfg=cfg)],
            cfg=cfg)

        # when
        actual_selected = classifiers.select_classifier_to_delete(
            randomfunc=RandomMock([0.5, 0.1, 0.5, 0.1, 0.1, 0.5]))

        # then
        assert much_worse == actual_selected

    def test_delete_a_classifier_delete(self, cfg):
        # given
        cl_1 = ACS2Classifier(action=1, cfg=cfg)
        cl_2 = ACS2Classifier(action=2, cfg=cfg)
        cl_3 = ACS2Classifier(action=3, cfg=cfg)
        cl_4 = ACS2Classifier(action=4, cfg=cfg)
        action_set = ClassifiersList([cl_1, cl_2], cfg=cfg)
        match_set = ClassifiersList([cl_2], cfg=cfg)
        population = ClassifiersList([cl_1, cl_2, cl_3, cl_4], cfg=cfg)

        # when
        action_set.delete_a_classifier(
            match_set, population, randomfunc=RandomMock([0.5, 0.1, 0.5, 0.5]))

        # then
        assert ClassifiersList([cl_1], cfg=cfg) == action_set
        assert ClassifiersList([], cfg=cfg) == match_set
        assert ClassifiersList([cl_1, cl_3, cl_4], cfg=cfg) == population

    def test_delete_a_classifier_decrease_numerosity(self, cfg):
        # given
        cl_1 = ACS2Classifier(action=1, cfg=cfg)
        cl_2 = ACS2Classifier(action=2, numerosity=3, cfg=cfg)
        cl_3 = ACS2Classifier(action=3, cfg=cfg)
        cl_4 = ACS2Classifier(action=4, cfg=cfg)
        action_set = ClassifiersList([cl_1, cl_2], cfg=cfg)
        match_set = ClassifiersList([cl_2], cfg=cfg)
        population = ClassifiersList([cl_1, cl_2, cl_3, cl_4], cfg=cfg)

        # when
        action_set.delete_a_classifier(
            match_set, population, randomfunc=RandomMock([0.5, 0.1, 0.5, 0.5]))

        expected_action_set = ClassifiersList(
            [cl_1, ACS2Classifier(action=2, numerosity=2, cfg=cfg)],
            cfg=cfg)
        expected_match_set = ClassifiersList(
            [ACS2Classifier(action=2, numerosity=2, cfg=cfg)],
            cfg=cfg)
        expected_population = ClassifiersList(
            [cl_1, ACS2Classifier(action=2, numerosity=2, cfg=cfg),
             cl_3, cl_4],
            cfg=cfg)

        # then
        assert expected_action_set == action_set
        assert expected_match_set == match_set
        assert expected_population == population

    def test_delete_ga_classifiers(self, cfg):
        # given
        cl_1 = ACS2Classifier(action=1, cfg=cfg)
        cl_2 = ACS2Classifier(action=2, numerosity=20, cfg=cfg)
        cl_3 = ACS2Classifier(action=3, cfg=cfg)
        cl_4 = ACS2Classifier(action=4, cfg=cfg)
        action_set = ClassifiersList([cl_1, cl_2], cfg=cfg)
        match_set = ClassifiersList([cl_2], cfg=cfg)
        population = ClassifiersList([cl_1, cl_2, cl_3, cl_4], cfg=cfg)

        # when
        action_set.delete_ga_classifiers(
            population, match_set, 2,
            randomfunc=RandomMock(([0.5, 0.1] + [0.5] * 19) * 3))

        expected_action_set = ClassifiersList(
            [cl_1, ACS2Classifier(action=2, numerosity=17, cfg=cfg)],
            cfg=cfg)
        expected_match_set = ClassifiersList(
            [ACS2Classifier(action=2, numerosity=17, cfg=cfg)],
            cfg=cfg)
        expected_population = ClassifiersList(
            [cl_1, ACS2Classifier(action=2, numerosity=17, cfg=cfg),
             cl_3, cl_4], cfg=cfg)

        # then
        assert expected_action_set == action_set
        assert expected_match_set == match_set
        assert expected_population == population

    def test_other_preferred_to_delete_if_significantly_worse(self, cfg):
        # given
        cl = ACS2Classifier(quality=0.5, cfg=cfg)
        cl_del = ACS2Classifier(quality=0.8, cfg=cfg)

        # when
        selected_cl = ClassifiersList(cfg=cfg)\
            .select_preferred_to_delete(cl, cl_del)

        # then
        assert cl == selected_cl

    def test_other_not_preferred_to_delete_if_significantly_better(self, cfg):
        # given
        cl = ACS2Classifier(quality=0.8, cfg=cfg)
        cl_del = ACS2Classifier(quality=0.5, cfg=cfg)

        # when
        selected_cl = ClassifiersList(cfg=cfg)\
            .select_preferred_to_delete(cl, cl_del)

        # then
        assert cl_del == selected_cl

    def test_if_selected_somewhat_close_to_other_marked_considered1(self, cfg):
        # given
        cl = ACS2Classifier(quality=0.8, cfg=cfg)
        cl.mark[0] = '0'
        cl_del = ACS2Classifier(quality=0.85, cfg=cfg)

        # when
        selected_cl = ClassifiersList(cfg=cfg)\
            .select_preferred_to_delete(cl, cl_del)

        # then
        assert cl_del.is_marked() is False
        assert cl.is_marked() is True
        assert cl == selected_cl

    def test_if_selected_somewhat_close_to_other_marked_considered2(self, cfg):
        # given
        cl = ACS2Classifier(quality=0.8, cfg=cfg)
        cl_del = ACS2Classifier(quality=0.85, cfg=cfg)
        cl_del.mark[0] = '0'

        # when
        selected_cl = ClassifiersList(cfg=cfg)\
            .select_preferred_to_delete(cl, cl_del)

        # then
        assert cl.is_unmarked() is True
        assert cl_del.is_marked() is True
        assert cl_del == selected_cl

    def test_if_selected_close_to_other_both_umarked_tav_considered1(self,
                                                                     cfg):
        # given
        cl = ACS2Classifier(quality=0.8, tav=0.2, cfg=cfg)
        cl_del = ACS2Classifier(quality=0.85, tav=0.1, cfg=cfg)

        # when
        selected_cl = ClassifiersList(cfg=cfg)\
            .select_preferred_to_delete(cl, cl_del)

        # then
        assert cl.is_marked() is False
        assert cl_del.is_marked() is False
        assert cl == selected_cl

    def test_if_selected_close_to_other_both_umarked_tav_considered2(self,
                                                                     cfg):
        # given
        cl = ACS2Classifier(quality=0.8, tav=0.1, cfg=cfg)
        cl_del = ACS2Classifier(quality=0.85, tav=0.1, cfg=cfg)

        # when
        selected_cl = ClassifiersList(cfg=cfg)\
            .select_preferred_to_delete(cl, cl_del)

        # then
        assert cl.is_marked() is False
        assert cl_del.is_marked() is False
        assert cl_del == selected_cl

    def test_if_selected_close_to_other_both_marked_tav_considered1(self, cfg):
        # given
        cl = ACS2Classifier(quality=0.85, tav=0.2, cfg=cfg)
        cl.mark[0] = '0'
        cl_del = ACS2Classifier(quality=0.8, tav=0.1, cfg=cfg)
        cl_del.mark[0] = '0'

        # when
        selected_cl = ClassifiersList(cfg=cfg)\
            .select_preferred_to_delete(cl, cl_del)

        # then
        assert cl.is_marked() is True
        assert cl_del.is_marked() is True
        assert cl == selected_cl

    def test_if_selected_close_to_other_both_marked_tav_considered2(self, cfg):
        # given
        cl = ACS2Classifier(quality=0.8, tav=0.1, cfg=cfg)
        cl.mark[0] = '0'
        cl_del = ACS2Classifier(quality=0.85, tav=0.1, cfg=cfg)
        cl_del.mark[0] = '0'

        # when
        selected_cl = ClassifiersList(cfg=cfg)\
            .select_preferred_to_delete(cl, cl_del)

        # then
        assert cl.is_marked() is True
        assert cl_del.is_marked() is True
        assert cl_del == selected_cl

    def test_overall_numerosity(self, cfg):
        population = ClassifiersList(cfg=cfg)
        assert 0 == population.overall_numerosity()

        population.append(ACS2Classifier(numerosity=2, cfg=cfg))
        assert 2 == population.overall_numerosity()

        population.append(ACS2Classifier(numerosity=1, cfg=cfg))
        assert 3 == population.overall_numerosity()

        population.append(ACS2Classifier(numerosity=3, cfg=cfg))
        assert 6 == population.overall_numerosity()

    def test_add_ga_classifier_add(self, cfg):
        # given
        cl_1 = ACS2Classifier(action=1, cfg=cfg)
        cl_2 = ACS2Classifier(action=2,
                              condition='1#######',
                              cfg=cfg)
        cl_3 = ACS2Classifier(action=3, cfg=cfg)
        cl_4 = ACS2Classifier(action=4, cfg=cfg)
        action_set = ClassifiersList([cl_1], cfg=cfg)
        match_set = ClassifiersList([], cfg=cfg)
        population = ClassifiersList([cl_1, cl_3, cl_4], cfg)

        # when
        action_set.add_ga_classifier(cl_2, match_set, population)

        # then
        assert ClassifiersList([cl_2], cfg=cfg) == match_set
        assert ClassifiersList([cl_1, cl_3, cl_4, cl_2], cfg) == population

    def test_add_ga_classifier_increase_numerosity(self, cfg):
        # given
        cl_1 = ACS2Classifier(action=2,
                              condition='1#######',
                              cfg=cfg)
        cl_2 = ACS2Classifier(action=2,
                              condition='1#######',
                              cfg=cfg)
        cl_3 = ACS2Classifier(action=3, cfg=cfg)
        cl_4 = ACS2Classifier(action=4, cfg=cfg)

        action_set = ClassifiersList([cl_1], cfg)
        match_set = ClassifiersList([cl_1], cfg)
        population = ClassifiersList([cl_1, cl_3, cl_4], cfg)

        # when
        action_set.add_ga_classifier(cl_2, match_set, population)
        new_classifier = ACS2Classifier(
            action=2,
            condition=Condition('1#######', cfg=cfg),
            numerosity=2,
            cfg=cfg)

        # then
        assert ClassifiersList([new_classifier, cl_3, cl_4], cfg) == population

    @pytest.mark.skip(reason="todo: test with deterministic RNG")
    def test_apply_ga(self, cfg):
        # given
        cl_1 = ACS2Classifier(
            condition='#1#1#1#1',
            numerosity=12,
            cfg=cfg)
        cl_2 = ACS2Classifier(
            condition='0#0#0#0#',
            numerosity=9,
            cfg=cfg)
        action_set = ClassifiersList([cl_1, cl_2], cfg=cfg)
        match_set = ClassifiersList([cl_1, cl_2], cfg=cfg)
        population = ClassifiersList([cl_1, cl_2], cfg=cfg)

        random_sequence = \
            [
                0.1, 0.6,  # parent selection
                0.1, 0.5, 0.5, 0.5,  # mutation of child1
                0.5, 0.1, 0.5, 0.5,  # mutation of child2
                0.1,  # do crossover
            ] + [0.5] * 12 + [0.2] + [0.5] * 8 + \
            [0.2] + [0.5] * 20 + [0.2] + [0.5] * 20

        # when
        action_set.apply_ga(101, population, match_set, None,
                            randomfunc=RandomMock(random_sequence),
                            samplefunc=SampleMock([0, 4]))

        # then
        modified_parent1 = ACS2Classifier(condition='#1#1#1#1',
                                          numerosity=10,
                                          tga=101,
                                          cfg=cfg)

        modified_parent2 = ACS2Classifier(condition='0#0#0#0#',
                                          numerosity=8,
                                          tga=101,
                                          cfg=cfg)

        child1 = ACS2Classifier(condition='0####1#1',
                                quality=0.25,
                                talp=101,
                                tga=101,
                                cfg=cfg)

        child2 = ACS2Classifier(condition='###10#0#',
                                quality=0.25,
                                talp=101,
                                tga=101,
                                cfg=cfg)

        expected_population = ClassifiersList(
            [modified_parent1, modified_parent2, child1, child2], cfg)

        # it might sometime fails because one function RNDG is not mocked
        assert expected_population == population
        assert expected_population == match_set
        assert expected_population == action_set

    def test_should_insert_classifier_1(self, cfg):
        population = ClassifiersList(cfg=cfg)

        with pytest.raises(TypeError):
            # Try to insert an integer instead of classifier object
            population.append(4)

    def test_should_insert_classifier_2(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)

        # when
        population.append(ACS2Classifier(cfg=cfg))

        # then
        assert 1 == len(population)

    def test_should_form_match_set(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)
        situation = Perception('11110000')

        # C1 - general condition
        c1 = ACS2Classifier(cfg=cfg)

        # C2 - matching condition
        c2 = ACS2Classifier(condition='1###0###', cfg=cfg)

        # C3 - non-matching condition
        c3 = ACS2Classifier(condition='0###1###', cfg=cfg)

        population.append(c1)
        population.append(c2)
        population.append(c3)

        # when
        match_set = ClassifiersList.form_match_set(population,
                                                   situation,
                                                   cfg=cfg)
        # then
        assert 2 == len(match_set)
        assert c1 in match_set
        assert c2 in match_set

    def test_should_form_action_set(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)
        c0 = ACS2Classifier(action=0, cfg=cfg)
        c01 = ACS2Classifier(action=0, cfg=cfg)
        c1 = ACS2Classifier(action=1, cfg=cfg)

        population.append(c0)
        population.append(c01)
        population.append(c1)

        # when & then
        action_set = ClassifiersList.form_action_set(population, 0, cfg)
        assert 2 == len(action_set)
        assert c0 in action_set
        assert c01 in action_set

        # when & then
        action_set = ClassifiersList.form_action_set(population, 1, cfg)
        assert 1 == len(action_set)
        assert c1 in action_set

    def test_should_expand(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)
        c0 = ACS2Classifier(action=0, cfg=cfg)
        c1 = ACS2Classifier(action=1, numerosity=2, cfg=cfg)
        c2 = ACS2Classifier(action=2, numerosity=3, cfg=cfg)

        population.append(c0)
        population.append(c1)
        population.append(c2)

        # when
        expanded = population.expand()

        # then
        assert 6 == len(expanded)
        assert c0 in expanded
        assert c1 in expanded
        assert c2 in expanded

    def test_should_calculate_maximum_fitness(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)

        # when & then
        # C1 - does not anticipate change
        c1 = ACS2Classifier(cfg=cfg)
        population.append(c1)
        assert 0.0 == population.get_maximum_fitness()

        # when & then
        # C2 - does anticipate some change
        c2 = ACS2Classifier(effect='1###0###',
                            reward=0.25,
                            cfg=cfg)
        population.append(c2)
        assert 0.125 == population.get_maximum_fitness()

        # when & then
        # C3 - does anticipate change and is quite good
        c3 = ACS2Classifier(effect='1#######',
                            quality=0.8,
                            reward=5,
                            cfg=cfg)
        population.append(c3)
        assert 4 == population.get_maximum_fitness()

    def test_should_return_all_possible_actions(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)
        actions = set()

        # when
        for _ in range(1000):
            act = population.choose_action(epsilon=1.0)
            actions.add(act)

        # then
        assert 8 == len(actions)

    def test_should_get_similar_classifier(self, cfg):
        # given
        pop = ClassifiersList(cfg=cfg)
        pop.append(ACS2Classifier(action=1, cfg=cfg))
        pop.append(ACS2Classifier(action=2, cfg=cfg))
        pop.append(ACS2Classifier(action=3, cfg=cfg))

        # when & then
        # No similar classifiers exist
        assert pop.get_similar(ACS2Classifier(action=4, cfg=cfg)) is None

        # when & then
        # Should find similar classifier
        assert pop.get_similar(ACS2Classifier(action=2, cfg=cfg)) is not None

    def test_should_apply_reinforcement_learning(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)
        c1 = ACS2Classifier(cfg=cfg)
        c1.r = 34.29
        c1.ir = 11.29
        population.append(c1)

        # when
        population.apply_reinforcement_learning(0, 28.79)

        # then
        assert abs(33.94 - population[0].r) < 0.1
        assert abs(10.74 - population[0].ir) < 0.1

    def test_should_insert_alp_offspring_1(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)
        new_list = ClassifiersList(cfg=cfg)

        child = ACS2Classifier(
            condition='1##1#010',
            action=0,
            effect='0####101',
            quality=0.5,
            reward=8.96245,
            intermediate_reward=0,
            experience=1,
            tga=423,
            talp=423,
            tav=27.3182,
            cfg=cfg
        )

        c1 = ACS2Classifier(
            condition='1##1#010',
            action=0,
            effect='0####101',
            quality=0.571313,
            reward=7.67011,
            intermediate_reward=0,
            experience=3,
            tga=225,
            talp=423,
            tav=70.881,
            cfg=cfg
        )

        c2 = ACS2Classifier(
            condition='1####010',
            action=0,
            effect='0####101',
            quality=0.462151,
            reward=8.96245,
            intermediate_reward=0,
            experience=11,
            tga=143,
            talp=423,
            tav=27.3182,
            cfg=cfg
        )

        c3 = ACS2Classifier(
            condition='1####0##',
            action=0,
            effect='0####1##',
            quality=0.31452,
            reward=9.04305,
            intermediate_reward=0,
            experience=19,
            tga=49,
            talp=423,
            tav=19.125,
            cfg=cfg
        )

        # Add classifiers into current ClassifierList
        population.extend([c1, c2, c3])

        # when
        population.add_alp_classifier(child, new_list)

        # then
        assert 3 == len(population)
        assert c1 in population
        assert c2 in population
        assert c3 in population
        assert abs(0.592747 - c1.q) < 0.01

    def test_should_insert_alp_offspring_2(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)
        new_list = ClassifiersList(cfg=cfg)

        child = ACS2Classifier(
            condition='#1O##O##',
            action=0,
            quality=0.5,
            reward=18.206,
            intermediate_reward=0,
            experience=1,
            tga=747,
            talp=747,
            tav=22.0755,
            cfg=cfg
        )

        c1 = ACS2Classifier(
            condition='#1O#O###',
            action=0,
            quality=0.650831,
            reward=14.8323,
            intermediate_reward=0,
            experience=5,
            tga=531,
            talp=747,
            tav=48.3562,
            cfg=cfg
        )

        c2 = ACS2Classifier(
            condition='##O#O###',
            action=0,
            quality=0.79094,
            reward=9.97782,
            intermediate_reward=0,
            experience=10,
            tga=330,
            talp=747,
            tav=43.7171,
            cfg=cfg
        )

        c3 = ACS2Classifier(
            condition='#1O###1O',
            action=0,
            effect='#O1####1',
            quality=0.515369,
            reward=8.3284,
            intermediate_reward=0,
            experience=8,
            tga=316,
            talp=747,
            tav=57.8883,
            cfg=cfg
        )

        c3.mark[0].update(['1'])
        c3.mark[3].update(['0'])
        c3.mark[4].update(['0'])
        c3.mark[5].update(['0'])

        c4 = ACS2Classifier(
            condition='####O###',
            action=0,
            quality=0.903144,
            reward=14.8722,
            intermediate_reward=0,
            experience=25,
            tga=187,
            talp=747,
            tav=23.0038,
            cfg=cfg
        )

        c5 = ACS2Classifier(
            condition='#1O####O',
            action=0,
            effect='#O1####1',
            quality=0.647915,
            reward=9.24712,
            intermediate_reward=0,
            experience=14,
            tga=154,
            talp=747,
            tav=44.5457,
            cfg=cfg
        )
        c5.mark[0].update(['1'])
        c5.mark[3].update(['0', '1'])
        c5.mark[4].update(['0', '1'])
        c5.mark[5].update(['0', '1'])
        c5.mark[6].update(['0', '1'])

        c6 = ACS2Classifier(
            condition='#1O#####',
            action=0,
            quality=0.179243,
            reward=18.206,
            intermediate_reward=0,
            experience=29,
            tga=104,
            talp=747,
            tav=22.0755,
            cfg=cfg
        )
        c6.mark[0].update(['1'])
        c6.mark[3].update(['1'])
        c6.mark[4].update(['1'])
        c6.mark[5].update(['1'])
        c6.mark[6].update(['0', '1'])
        c6.mark[7].update(['0', '1'])

        c7 = ACS2Classifier(
            condition='##O#####',
            action=0,
            quality=0.100984,
            reward=15.91,
            intermediate_reward=0,
            experience=44,
            tga=58,
            talp=747,
            tav=14.4171,
            cfg=cfg
        )
        c7.mark[0].update(['0', '1'])
        c7.mark[1].update(['0', '1'])
        c7.mark[3].update(['0', '1'])
        c7.mark[5].update(['0', '1'])
        c7.mark[6].update(['0', '1'])
        c7.mark[7].update(['0', '1'])

        # Add classifiers into current ClassifierList
        population.extend([c1, c2, c3, c4, c5, c6, c7])

        # When
        population.add_alp_classifier(child, new_list)

        # Then
        assert 7 == len(population)
        assert 0 == len(new_list)
        assert c1 in population
        assert c2 in population
        assert c3 in population
        assert c4 in population
        assert c5 in population
        assert c6 in population
        assert c7 in population

        # `C4` should be subsumer of `child`
        assert abs(0.907987 - c4.q) < 0.01
