import unittest
from random import randint

from alcs.acs2 import ClassifiersList, Condition, Effect, Classifier

from alcs import Perception
from alcs.acs2.testrandom import TestRandom, TestSample


class ClassifierListTest(unittest.TestCase):
    def setUp(self):
        self.population = ClassifiersList()

    def test_find_subsumer_finds_single_subsumer(self):
        subsumer = Classifier(condition=Condition('###0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                              reward=1.35, experience=23)
        nonsubsumer = Classifier()

        classifier = Classifier(condition=Condition('1##0####'), action=3, effect=Effect('##1#####'), quality=0.5,
                                reward=0.35, experience=1)
        classifiers_list = ClassifiersList([nonsubsumer, subsumer, nonsubsumer])
        actual_subsumer = ClassifiersList.find_subsumer(classifier, classifiers_list, choice_func=lambda l: l[0])
        self.assertEqual(subsumer, actual_subsumer)

    def test_find_subsumer_finds_single_subsumer_among_nonsubsumers(self):
        subsumer = Classifier(condition=Condition('###0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                              reward=1.35, experience=23)
        nonsubsumer = Classifier()

        classifier = Classifier(condition=Condition('1##0####'), action=3, effect=Effect('##1#####'), quality=0.5,
                                reward=0.35, experience=1)
        classifiers_list = ClassifiersList([nonsubsumer, subsumer, nonsubsumer])

        actual_subsumer = ClassifiersList.find_subsumer(classifier, classifiers_list, choice_func=lambda l: l[0])
        self.assertEqual(actual_subsumer, subsumer)

    def test_find_subsumer_finds_selects_more_general_subsumer1(self):
        subsumer1 = Classifier(condition=Condition('1##0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                               reward=1.35, experience=23)
        subsumer2 = Classifier(condition=Condition('###0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                               reward=1.35, experience=23)
        nonsubsumer = Classifier()

        classifier = Classifier(condition=Condition('11#0####'), action=3, effect=Effect('##1#####'), quality=0.5,
                                reward=0.35, experience=1)
        classifiers_list = ClassifiersList([nonsubsumer, subsumer2, subsumer1, nonsubsumer])

        actual_subsumer = ClassifiersList.find_subsumer(classifier, classifiers_list, choice_func=lambda l: l[0])
        self.assertEqual(actual_subsumer, subsumer2)

    def test_find_subsumer_finds_selects_more_general_subsumer2(self):
        subsumer1 = Classifier(condition=Condition('1##0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                               reward=1.35, experience=23)
        subsumer2 = Classifier(condition=Condition('###0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                               reward=1.35, experience=23)
        nonsubsumer = Classifier()

        classifier = Classifier(condition=Condition('11#0####'), action=3, effect=Effect('##1#####'), quality=0.5,
                                reward=0.35, experience=1)
        classifiers_list = ClassifiersList([nonsubsumer, subsumer1, subsumer2, nonsubsumer])

        actual_subsumer = ClassifiersList.find_subsumer(classifier, classifiers_list, choice_func=lambda l: l[0])
        self.assertEqual(actual_subsumer, subsumer2)

    def test_find_subsumer_finds_randomly_selects_one_of_equally_general_subsumers(self):
        subsumer1 = Classifier(condition=Condition('1##0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                               reward=1.35, experience=23)
        subsumer2 = Classifier(condition=Condition('#1#0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                               reward=1.35, experience=23)
        nonsubsumer = Classifier()

        classifier = Classifier(condition=Condition('11#0####'), action=3, effect=Effect('##1#####'), quality=0.5,
                                reward=0.35, experience=1)
        classifiers_list = ClassifiersList([nonsubsumer, subsumer1, subsumer2, nonsubsumer])

        actual_subsumer = ClassifiersList.find_subsumer(classifier, classifiers_list, choice_func=lambda l: l[0])
        self.assertEqual(actual_subsumer, subsumer1)
        actual_subsumer = ClassifiersList.find_subsumer(classifier, classifiers_list, choice_func=lambda l: l[1])
        self.assertEqual(actual_subsumer, subsumer2)

    def test_find_subsumer_selects_most_general_subsumer(self):
        subsumer1 = Classifier(condition=Condition('1##0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                               reward=1.35, experience=23)
        subsumer2 = Classifier(condition=Condition('#1#0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                               reward=1.35, experience=23)
        most_general = Classifier(condition=Condition('###0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                                  reward=1.35, experience=23)
        nonsubsumer = Classifier()

        classifier = Classifier(condition=Condition('11#0####'), action=3, effect=Effect('##1#####'), quality=0.5,
                                reward=0.35, experience=1)
        classifiers_list = ClassifiersList([nonsubsumer, subsumer1, nonsubsumer, most_general, subsumer2, nonsubsumer])

        actual_subsumer = ClassifiersList.find_subsumer(classifier, classifiers_list, choice_func=lambda l: l[0])
        self.assertEqual(actual_subsumer, most_general)

    def test_find_old_classifier_only_subsumer(self):
        subsumer1 = Classifier(condition=Condition('1##0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                               reward=1.35, experience=23)
        subsumer2 = Classifier(condition=Condition('#1#0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                               reward=1.35, experience=23)
        most_general = Classifier(condition=Condition('###0####'), action=3, effect=Effect('##1#####'), quality=0.93,
                                  reward=1.35, experience=23)
        nonsubsumer = Classifier()

        classifier = Classifier(condition=Condition('11#0####'), action=3, effect=Effect('##1#####'), quality=0.5,
                                reward=0.35, experience=1)
        classifiers_list = ClassifiersList([nonsubsumer, subsumer1, nonsubsumer, most_general, subsumer2, nonsubsumer])

        actual_old_classifier = ClassifiersList.find_old_classifier(classifier, classifiers_list)
        self.assertEqual(most_general, actual_old_classifier)

    def test_find_old_classifier_only_similar(self):
        classifier_1 = Classifier(action=1, experience=32)
        classifier_2 = Classifier(action=1)
        classifiers = ClassifiersList([classifier_1, Classifier(action=2), Classifier(action=3), classifier_2])
        self.assertEqual(classifier_1, ClassifiersList.find_old_classifier(Classifier(action=1), classifiers))

    def test_find_old_classifier_similar_and_subsumer_subsumer_returned(self):
        subsumer = Classifier(condition=Condition('1#######'), action=1, experience=21, quality=0.95)
        similar = Classifier(condition=Condition('10######'), action=1)

        existing_classifiers = ClassifiersList([similar, subsumer])
        classifier = Classifier(condition=Condition('10######'), action=1)
        self.assertTrue(subsumer.does_subsume(classifier))
        self.assertTrue(similar.is_similar(classifier))

        self.assertEqual(subsumer, ClassifiersList.find_old_classifier(classifier, existing_classifiers))

    def test_find_old_classifier_none(self):
        self.assertIsNone(ClassifiersList.find_old_classifier(Classifier(), None))
        self.assertIsNone(ClassifiersList.find_old_classifier(Classifier(), ClassifiersList([])))

    def test_select_classifier_to_delete(self):
        selected_first = Classifier(quality=0.5)
        much_worse = Classifier(quality=0.2)
        yet_another_to_consider = Classifier(quality=0.2)
        classifiers = ClassifiersList([Classifier(), selected_first, Classifier(), much_worse,
                                       yet_another_to_consider, Classifier()])
        actual_selected = classifiers.select_classifier_to_delete(randomfunc=TestRandom([0.5, 0.1, 0.5, 0.1, 0.1, 0.5]))
        self.assertEqual(much_worse, actual_selected)

    def test_delete_a_classifier_delete(self):
        cl_1 = Classifier(action=1)
        cl_2 = Classifier(action=2)
        cl_3 = Classifier(action=3)
        cl_4 = Classifier(action=4)
        action_set = ClassifiersList([cl_1, cl_2])
        match_set = ClassifiersList([cl_2])
        population = ClassifiersList([cl_1, cl_2, cl_3, cl_4])
        action_set.delete_a_classifier(match_set, population, randomfunc=TestRandom([0.5, 0.1, 0.5, 0.5]))
        self.assertEqual(ClassifiersList([cl_1]), action_set)
        self.assertEqual(ClassifiersList([]), match_set)
        self.assertEqual(ClassifiersList([cl_1, cl_3, cl_4]), population)

    def test_delete_a_classifier_decrease_numerosity(self):
        cl_1 = Classifier(action=1)
        cl_2 = Classifier(action=2, numerosity=3)
        cl_3 = Classifier(action=3)
        cl_4 = Classifier(action=4)
        action_set = ClassifiersList([cl_1, cl_2])
        match_set = ClassifiersList([cl_2])
        population = ClassifiersList([cl_1, cl_2, cl_3, cl_4])
        action_set.delete_a_classifier(match_set, population, randomfunc=TestRandom([0.5, 0.1, 0.5, 0.5]))

        self.assertListEqual(ClassifiersList([cl_1, Classifier(action=2, numerosity=2)]), action_set)
        self.assertListEqual(ClassifiersList([Classifier(action=2, numerosity=2)]), match_set)
        self.assertListEqual(ClassifiersList([cl_1, Classifier(action=2, numerosity=2), cl_3, cl_4]), population)

    def test_delete_ga_classifiers(self):
        cl_1 = Classifier(action=1)
        cl_2 = Classifier(action=2, numerosity=20)
        cl_3 = Classifier(action=3)
        cl_4 = Classifier(action=4)
        action_set = ClassifiersList([cl_1, cl_2])
        match_set = ClassifiersList([cl_2])
        population = ClassifiersList([cl_1, cl_2, cl_3, cl_4])
        action_set.delete_ga_classifiers(population, match_set, 2, randomfunc=TestRandom(([0.5, 0.1] + [0.5] * 19) * 3))

        self.assertListEqual(ClassifiersList([cl_1, Classifier(action=2, numerosity=17)]), action_set)
        self.assertListEqual(ClassifiersList([Classifier(action=2, numerosity=17)]), match_set)
        self.assertListEqual(ClassifiersList([cl_1, Classifier(action=2, numerosity=17), cl_3, cl_4]), population)

    def test_other_preferred_to_delete_if_significantly_worse(self):
        cl = Classifier(quality=0.5)
        cl_del = Classifier(quality=0.8)
        self.assertEqual(cl, ClassifiersList().select_preferred_to_delete(cl, cl_del))

    def test_other_not_preferred_to_delete_if_significantly_better(self):
        cl = Classifier(quality=0.8)
        cl_del = Classifier(quality=0.5)
        self.assertEqual(cl_del, ClassifiersList().select_preferred_to_delete(cl, cl_del))

    def test_if_selected_somewhat_close_to_other_marked_considered1(self):
        cl = Classifier(quality=0.8)
        cl.mark[0] = '0'
        cl_del = Classifier(quality=0.85)
        self.assertFalse(cl_del.is_marked())
        self.assertTrue(cl.is_marked())
        self.assertEqual(cl, ClassifiersList().select_preferred_to_delete(cl, cl_del))

    def test_if_selected_somewhat_close_to_other_marked_considered2(self):
        cl = Classifier(quality=0.8)
        cl_del = Classifier(quality=0.85)
        cl_del.mark[0] = '0'
        self.assertTrue(cl.is_unmarked())
        self.assertTrue(cl_del.is_marked())
        self.assertEqual(cl_del, ClassifiersList().select_preferred_to_delete(cl, cl_del))

    def test_if_selected_somewhat_close_to_other_both_umarked_tav_considered1(self):
        cl = Classifier(quality=0.8, tav=0.2)
        cl_del = Classifier(quality=0.85, tav=0.1)
        self.assertFalse(cl.is_marked())
        self.assertFalse(cl_del.is_marked())
        self.assertEqual(cl, ClassifiersList().select_preferred_to_delete(cl, cl_del))

    def test_if_selected_somewhat_close_to_other_both_umarked_tav_considered2(self):
        cl = Classifier(quality=0.8, tav=0.1)
        cl_del = Classifier(quality=0.85, tav=0.1)
        self.assertFalse(cl.is_marked())
        self.assertFalse(cl_del.is_marked())
        self.assertEqual(cl_del, ClassifiersList().select_preferred_to_delete(cl, cl_del))

    def test_if_selected_somewhat_close_to_other_both_marked_tav_considered1(self):
        cl = Classifier(quality=0.85, tav=0.2)
        cl.mark[0] = '0'
        cl_del = Classifier(quality=0.8, tav=0.1)
        cl_del.mark[0] = '0'
        self.assertTrue(cl.is_marked())
        self.assertTrue(cl_del.is_marked())
        self.assertEqual(cl, ClassifiersList().select_preferred_to_delete(cl, cl_del))

    def test_if_selected_somewhat_close_to_other_both_marked_tav_considered2(self):
        cl = Classifier(quality=0.8, tav=0.1)
        cl.mark[0] = '0'
        cl_del = Classifier(quality=0.85, tav=0.1)
        cl_del.mark[0] = '0'
        self.assertTrue(cl.is_marked())
        self.assertTrue(cl_del.is_marked())
        self.assertEqual(cl_del, ClassifiersList().select_preferred_to_delete(cl, cl_del))

    def test_overall_numerosity(self):
        population = ClassifiersList()
        self.assertEqual(0, population.overall_numerosity())

        population.append(Classifier(numerosity=2))
        self.assertEqual(2, population.overall_numerosity())
        population.append(Classifier(numerosity=1))
        self.assertEqual(3, population.overall_numerosity())
        population.append(Classifier(numerosity=3))
        self.assertEqual(6, population.overall_numerosity())

    def test_add_ga_classifier_add(self):
        cl_1 = Classifier(action=1)
        cl_2 = Classifier(action=2, condition=Condition('1#######'))
        cl_3 = Classifier(action=3)
        cl_4 = Classifier(action=4)
        action_set = ClassifiersList([cl_1])
        match_set = ClassifiersList([])
        population = ClassifiersList([cl_1, cl_3, cl_4])
        action_set.add_ga_classifier(cl_2, match_set, population)
        # self.assertEqual(ClassifiersList([cl_1, cl_2]), action_set)
        self.assertEqual(ClassifiersList([cl_2]), match_set)
        self.assertEqual(ClassifiersList([cl_1, cl_3, cl_4, cl_2]), population)

    def test_add_ga_classifier_increase_numerosity(self):
        cl_1 = Classifier(action=2, condition=Condition('1#######'))
        cl_2 = Classifier(action=2, condition=Condition('1#######'))
        cl_3 = Classifier(action=3)
        cl_4 = Classifier(action=4)
        action_set = ClassifiersList([cl_1])
        match_set = ClassifiersList([cl_1])
        population = ClassifiersList([cl_1, cl_3, cl_4])
        action_set.add_ga_classifier(cl_2, match_set, population)

        new_classifier = Classifier(action=2, condition=Condition('1#######'), numerosity=2)
        self.assertEqual(ClassifiersList([new_classifier, cl_3, cl_4]), population)

    def test_apply_ga(self):
        cl_1 = Classifier(Condition('#1#1#1#1'), numerosity=12)
        cl_2 = Classifier(Condition('0#0#0#0#'), numerosity=9)
        action_set = ClassifiersList([cl_1, cl_2])
        match_set = ClassifiersList([cl_1, cl_2])
        population = ClassifiersList([cl_1, cl_2])

        action_set.apply_ga(101, population, match_set, None, randomfunc=TestRandom([
                                                                                        0.1, 0.6,  # parent selection
                                                                                        0.1, 0.5, 0.5, 0.5,
                                                                                        # mutation of child1
                                                                                        0.5, 0.1, 0.5, 0.5,
                                                                                        # mutation of child2
                                                                                        0.1,  # do crossover
                                                                                    ] + [0.5] * 12 + [0.2] + [
                                                                                        0.5] * 8 + [0.2] + [
                                                                                        0.5] * 20 + [0.2] + [0.5] * 20
                                                                                    ), samplefunc=TestSample([0, 4]))

        modified_parent1 = Classifier(Condition('#1#1#1#1'), numerosity=10, tga=101)
        modified_parent2 = Classifier(Condition('0#0#0#0#'), numerosity=8, tga=101)
        child1 = Classifier(Condition('0####1#1'), quality=0.25, talp=101, tga=101)
        child2 = Classifier(Condition('###10#0#'), quality=0.25, talp=101, tga=101)
        expected_population = ClassifiersList([modified_parent1, modified_parent2, child1, child2])
        self.assertEqual(expected_population, population)
        self.assertEqual(expected_population, match_set)
        self.assertEqual(ClassifiersList([modified_parent1, modified_parent2]), action_set)

    def test_should_select_parents1(self):
        population = ClassifiersList()
        c0 = Classifier(Condition('######00'))
        c1 = Classifier(Condition('######01'))
        c2 = Classifier(Condition('######10'))
        c3 = Classifier(Condition('######11'))
        population.append(c0)
        population.append(c1)
        population.append(c2)
        population.append(c3)
        # q3num= 0.125 for all

        p1, p2 = population.select_parents(randomfunc=(TestRandom([0.7, 0.1])))
        self.assertEqual(c0, p1)
        # self.assertEqual(c2, p2)

        p1, p2 = population.select_parents(randomfunc=(TestRandom([0.3, 0.6])))
        self.assertEqual(c1, p1)
        self.assertEqual(c2, p2)

        p1, p2 = population.select_parents(randomfunc=(TestRandom([0.2, 0.8])))
        self.assertEqual(c0, p1)
        self.assertEqual(c3, p2)

    def test_quality_and_numerosity_influence_parent_selection(self):
        population = ClassifiersList()
        c0 = Classifier(Condition('######00'), quality=1, numerosity=1)
        c1 = Classifier(Condition('######01'))
        c2 = Classifier(Condition('######10'))
        population.append(c0)  # q3num = 1
        population.append(c1)  # q3num = 0.0625
        population.append(c2)  # q3num = 0.0625

        p1, p2 = population.select_parents(randomfunc=(TestRandom([0.888, 0.999])))
        self.assertEqual(c1, p1)
        self.assertEqual(c2, p2)

        p1, p2 = population.select_parents(randomfunc=(TestRandom([0.888, 0.777])))
        self.assertEqual(c0, p1)
        self.assertEqual(c1, p2)

    def test_should_insert_classifier_1(self):
        # Try to insert an integer instead of classifier object
        self.assertRaises(TypeError, self.population.append, 4)

    def test_should_insert_classifier_2(self):
        self.population.append(Classifier())
        self.assertEqual(1, len(self.population))

    def test_should_form_match_set(self):
        situation = Perception(['1', '1', '1', '1', '0', '0', '0', '0'])

        # C1 - general condition
        c1 = Classifier()

        # C2 - matching condition
        c2 = Classifier(condition=['1', '#', '#', '#', '0', '#', '#', '#'])

        # C3 - non-matching condition
        c3 = Classifier(condition=['0', '#', '#', '#', '1', '#', '#', '#'])

        self.population.append(c1)
        self.population.append(c2)
        self.population.append(c3)

        match_set = ClassifiersList.form_match_set(self.population, situation)

        self.assertEqual(2, len(match_set))
        self.assertIn(c1, match_set)
        self.assertIn(c2, match_set)

    def test_should_form_action_set(self):
        c0 = Classifier(action=0)
        c01 = Classifier(action=0)
        c1 = Classifier(action=1)

        self.population.append(c0)
        self.population.append(c01)
        self.population.append(c1)

        action_set = ClassifiersList.form_action_set(self.population, 0)
        self.assertEqual(2, len(action_set))
        self.assertIn(c0, action_set)
        self.assertIn(c01, action_set)

        action_set = ClassifiersList.form_action_set(self.population, 1)
        self.assertEqual(1, len(action_set))
        self.assertIn(c1, action_set)

    def test_should_expand(self):
        # given
        c0 = Classifier(action=0)
        c1 = Classifier(action=1, numerosity=2)
        c2 = Classifier(action=2, numerosity=3)

        self.population.append(c0)
        self.population.append(c1)
        self.population.append(c2)

        # when
        expanded = self.population.expand()

        # then
        self.assertEqual(6, len(expanded))
        self.assertIn(c0, expanded)
        self.assertIn(c1, expanded)
        self.assertIn(c2, expanded)

    def test_should_calculate_maximum_fitness(self):
        # C1 - does not anticipate change
        c1 = Classifier()

        self.population.append(c1)
        self.assertEqual(0.0, self.population.get_maximum_fitness())

        # C2 - does anticipate some change
        c2 = Classifier(effect=['1', '#', '#', '#', '0', '#', '#', '#'],
                        reward=0.25)

        self.population.append(c2)
        self.assertEqual(0.125, self.population.get_maximum_fitness())

        # C3 - does anticipate change and is quite good
        c3 = Classifier(effect=['1', '#', '#', '#', '#', '#', '#', '#'],
                        quality=0.8,
                        reward=5)

        self.population.append(c3)
        self.assertEqual(4, self.population.get_maximum_fitness())

    def test_should_return_all_possible_actions(self):
        # Given
        actions = set()

        # When
        for _ in range(1000):
            act = self.population.choose_action(epsilon=1.0)
            actions.add(act)

        # Then
        self.assertEqual(8, len(actions))

    def test_should_return_best_fitness_action(self):
        # C1 - does not anticipate change
        c1 = Classifier(action=1)
        self.population.append(c1)

        # Some random action should be selected here
        best_action = self.population.choose_best_fitness_action()
        self.assertIsNotNone(best_action)

        # C2 - does anticipate some change
        c2 = Classifier(action=2,
                        effect=['1', '#', '#', '#', '0', '#', '#', '#'],
                        reward=0.25)
        self.population.append(c2)

        # Here C2 action should be selected
        best_action = self.population.choose_best_fitness_action()
        self.assertEqual(2, best_action)

        # C3 - does anticipate change and is quite good
        c3 = Classifier(action=3,
                        effect=['1', '#', '#', '#', '#', '#', '#', '#'],
                        quality=0.8,
                        reward=5)
        self.population.append(c3)

        # Here C3 has the biggest fitness score
        best_action = self.population.choose_best_fitness_action()
        self.assertEqual(3, best_action)

    def test_should_return_random_action(self):
        random_actions = []

        for _ in range(0, 500):
            random_actions.append(self.population.choose_random_action())

        min_action = min(random_actions)
        max_action = max(random_actions)

        self.assertEqual(0, min_action)
        self.assertEqual(7, max_action)

    def test_should_return_latest_action(self):
        c0 = Classifier(action=0)
        c0.talp = 1

        self.population.append(c0)

        # Should return first action with no classifiers
        self.assertEqual(1, self.population.choose_latest_action())

        # Add rest of classifiers
        self.population.append(Classifier(action=3))
        self.population.append(Classifier(action=7))
        self.population.append(Classifier(action=5))
        self.population.append(Classifier(action=1))
        self.population.append(Classifier(action=4))
        self.population.append(Classifier(action=2))
        self.population.append(Classifier(action=6))

        # Assign each classifier random talp from certain range
        for cl in self.population:
            cl.talp = randint(70, 100)

        # But third classifier (action 7) will be the executed long time ago
        self.population[2].talp = randint(10, 20)
        self.assertEqual(7, self.population.choose_latest_action())

    def test_should_return_worst_quality_action(self):
        c0 = Classifier(action=0)
        self.population.append(c0)

        # Should return C1 (because it's first not mentioned)
        self.assertEqual(1,
                         self.population.choose_action_from_knowledge_array())

        # Add rest of classifiers
        c1 = Classifier(action=1, numerosity=31, quality=0.72)
        self.population.append(c1)

        c2 = Classifier(action=2, numerosity=2, quality=0.6)
        self.population.append(c2)

        c3 = Classifier(action=3, numerosity=2, quality=0.63)
        self.population.append(c3)

        c4 = Classifier(action=4, numerosity=7, quality=0.75)
        self.population.append(c4)

        c5 = Classifier(action=5, numerosity=1, quality=0.63)
        self.population.append(c5)

        c6 = Classifier(action=6, numerosity=6, quality=0.52)
        self.population.append(c6)

        c7 = Classifier(action=7, numerosity=10, quality=0.36)
        self.population.append(c7)

        # Classifier C7 should be the worst here
        self.assertEqual(7,
                         self.population.choose_action_from_knowledge_array())

    def test_should_get_similar_classifier(self):
        self.population.append(Classifier(action=1))
        self.population.append(Classifier(action=2))
        self.population.append(Classifier(action=3))

        # No similar classifiers exist
        self.assertIsNone(self.population.get_similar(Classifier(action=4)))

        # Should find similar classifier
        self.assertIsNotNone(self.population.get_similar(Classifier(action=2)))

    def test_should_apply_reinforcement_learning(self):
        c1 = Classifier()
        c1.r = 34.29
        c1.ir = 11.29
        self.population.append(c1)

        self.population.apply_reinforcement_learning(0, 28.79)

        self.assertAlmostEqual(33.94, self.population[0].r, 2)
        self.assertAlmostEqual(10.73, self.population[0].ir, 2)

    def test_should_insert_alp_offspring_1(self):
        # Given
        new_list = ClassifiersList()

        # *##*#O*O N O####*O* (########) q: 0.5	r: 8.96245	i: 0
        # exp: 1 tga: 423	talp: 423 tav: 27.3182 num: 1
        child = Classifier(
            condition=Condition(['1', '#', '#', '1', '#', '0', '1', '0']),
            action=0,
            effect=Effect(['0', '#', '#', '#', '#', '1', '0', '1']),
            quality=0.5,
            reward=8.96245,
            intermediate_reward=0,
            experience=1,
            tga=423,
            talp=423,
            tav=27.3182
        )

        # *##*#O*O N O####*O* (########) q: 0.571313	r: 7.67011	i: 0
        # exp: 3 tga: 225	talp: 423 tav: 70.881 num: 1
        c1 = Classifier(
            condition=Condition(['1', '#', '#', '1', '#', '0', '1', '0']),
            action=0,
            effect=Effect(['0', '#', '#', '#', '#', '1', '0', '1']),
            quality=0.571313,
            reward=7.67011,
            intermediate_reward=0,
            experience=3,
            tga=225,
            talp=423,
            tav=70.881
        )

        # *####O*O N O####*O* (#{*,O}{O,*}O{O,*}###) q: 0.462151
        # r: 8.96245	i: 0	exp: 11 tga: 143	talp: 423
        # tav: 27.3182 num: 1
        c2 = Classifier(
            condition=Condition(['1', '#', '#', '#', '#', '0', '1', '0']),
            action=0,
            effect=Effect(['0', '#', '#', '#', '#', '1', '0', '1']),
            quality=0.462151,
            reward=8.96245,
            intermediate_reward=0,
            experience=11,
            tga=143,
            talp=423,
            tav=27.3182
        )

        # *####O## N O####*## (#{O,*}{*,O}{O,*}{O,*}#{*,O}{O,*}) q: 0.31452
        # r: 9.04305	i: 0	exp: 19 tga: 49	talp: 423 tav: 19.125 num: 1
        c3 = Classifier(
            condition=Condition(['1', '#', '#', '#', '#', '0', '#', '#']),
            action=0,
            effect=Effect(['0', '#', '#', '#', '#', '1', '#', '#']),
            quality=0.31452,
            reward=9.04305,
            intermediate_reward=0,
            experience=19,
            tga=49,
            talp=423,
            tav=19.125
        )

        # Add classifiers into current ClassifierList
        self.population.extend([c1, c2, c3])

        # When
        self.population.add_alp_classifier(child, new_list)

        # Then
        self.assertEqual(3, len(self.population))
        self.assertIn(c1, self.population)
        self.assertIn(c2, self.population)
        self.assertIn(c3, self.population)
        self.assertAlmostEqual(0.592747, c1.q, places=1)

    def test_should_insert_alp_offspring_2(self):
        # Given
        new_list = ClassifiersList()

        # #*O##O## S ######## (########) q: 0.5	r: 18.206	i: 0
        # exp: 1 tga: 747	talp: 747 tav: 22.0755 num: 1
        child = Classifier(
            condition=Condition('#1O##O##'),
            action=0,
            quality=0.5,
            reward=18.206,
            intermediate_reward=0,
            experience=1,
            tga=747,
            talp=747,
            tav=22.0755
        )

        # #*O#O### S ######## (########) q: 0.650831	r: 14.8323	i: 0
        # exp: 5 tga: 531	talp: 747 tav: 48.3562 num: 1
        c1 = Classifier(
            condition=Condition('#1O#O###'),
            action=0,
            quality=0.650831,
            reward=14.8323,
            intermediate_reward=0,
            experience=5,
            tga=531,
            talp=747,
            tav=48.3562
        )

        # ##O#O### S ######## (########) q: 0.79094	r: 9.97782	i: 0
        # exp: 10 tga: 330	talp: 747 tav: 43.7171 num: 1
        c2 = Classifier(
            condition=Condition('##O#O###'),
            action=0,
            quality=0.79094,
            reward=9.97782,
            intermediate_reward=0,
            experience=10,
            tga=330,
            talp=747,
            tav=43.7171
        )

        # #*O###*O S #O*####* (*##OOO##) q: 0.515369	r: 8.3284	i: 0
        # exp: 8 tga: 316	talp: 747 tav: 57.8883 num: 1
        c3 = Classifier(
            condition=Condition('#1O###1O'),
            action=0,
            effect=Effect('#O1####1'),
            quality=0.515369,
            reward=8.3284,
            intermediate_reward=0,
            experience=8,
            tga=316,
            talp=747,
            tav=57.8883
        )
        # TODO p2: Maybe checking wheter marked sometimes does not work..
        # I mean type incompability
        c3.mark[0].update(['1'])
        c3.mark[3].update(['0'])
        c3.mark[4].update(['0'])
        c3.mark[5].update(['0'])

        # ####O### S ######## (########) q: 0.903144	r: 14.8722	i: 0
        # exp: 25 tga: 187	talp: 747 tav: 23.0038 num: 1
        c4 = Classifier(
            condition=Condition('####O###'),
            action=0,
            quality=0.903144,
            reward=14.8722,
            intermediate_reward=0,
            experience=25,
            tga=187,
            talp=747,
            tav=23.0038
        )

        # #*O####O S #O*####* (*##{*,O}{*,O}{*,O}{O,*}#) q: 0.647915
        # r: 9.24712	i: 0	exp: 14 tga: 154	talp: 747
        # tav: 44.5457 num: 1
        c5 = Classifier(
            condition=Condition('#1O####O'),
            action=0,
            effect=Effect('#O1####1'),
            quality=0.647915,
            reward=9.24712,
            intermediate_reward=0,
            experience=14,
            tga=154,
            talp=747,
            tav=44.5457
        )
        c5.mark[0].update(['1'])
        c5.mark[3].update(['0', '1'])
        c5.mark[4].update(['0', '1'])
        c5.mark[5].update(['0', '1'])
        c5.mark[6].update(['0', '1'])

        # #*O##### S ######## (*##***{*,O}{O,*}) q: 0.179243	r: 18.206
        # i: 0	exp: 29 tga: 104	talp: 747 tav: 22.0755 num: 1
        c6 = Classifier(
            condition=Condition('#1O#####'),
            action=0,
            quality=0.179243,
            reward=18.206,
            intermediate_reward=0,
            experience=29,
            tga=104,
            talp=747,
            tav=22.0755
        )
        c6.mark[0].update(['1'])
        c6.mark[3].update(['1'])
        c6.mark[4].update(['1'])
        c6.mark[5].update(['1'])
        c6.mark[6].update(['0', '1'])
        c6.mark[7].update(['0', '1'])

        # ##O##### S ######## ({*,O}{O,*}#{O,*}*{O,*}{*,O}{*,O}) q: 0.100984
        # r: 15.91	i: 0	exp: 44 tga: 58	talp: 747 tav: 14.4171 num: 1
        c7 = Classifier(
            condition=Condition('##O#####'),
            action=0,
            quality=0.100984,
            reward=15.91,
            intermediate_reward=0,
            experience=44,
            tga=58,
            talp=747,
            tav=14.4171
        )
        c7.mark[0].update(['0', '1'])
        c7.mark[1].update(['0', '1'])
        c7.mark[3].update(['0', '1'])
        c7.mark[5].update(['0', '1'])
        c7.mark[6].update(['0', '1'])
        c7.mark[7].update(['0', '1'])

        # Add classifiers into current ClassifierList
        self.population.extend([c1, c2, c3, c4, c5, c6, c7])

        # When
        self.population.add_alp_classifier(child, new_list)

        # Then
        self.assertEqual(7, len(self.population))
        self.assertEqual(0, len(new_list))
        self.assertIn(c1, self.population)
        self.assertIn(c2, self.population)
        self.assertIn(c3, self.population)
        self.assertIn(c4, self.population)
        self.assertIn(c5, self.population)
        self.assertIn(c6, self.population)
        self.assertIn(c7, self.population)

        # `C4` should be subsumer of `child`
        self.assertAlmostEqual(0.907987, c4.q, places=1)
