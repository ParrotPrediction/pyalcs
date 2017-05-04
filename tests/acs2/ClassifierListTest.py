import unittest

from alcs.agent.Perception import Perception
from alcs.agent.acs2 import ClassifiersList, Classifier, Condition, Effect

from random import randint


class ClassifierListTest(unittest.TestCase):

    def setUp(self):
        self.population = ClassifiersList()

    def test_restrict_to_only_classifiers(self):
        # Try to insert an integer instead of classifier object
        self.assertRaises(TypeError, self.population.append, 4)

    def test_try_to_insert_classifier(self):
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

    def test_should_calculate_maximum_fitness(self):
        # C1 - does not anticipate change
        c1 = Classifier()

        self.population.append(c1)
        self.assertEqual(0.0, self.population.get_maximum_fitness())

        # C2 - does anticipate some change
        c2 = Classifier(effect=['1', '#', '#', '#', '0', '#', '#', '#'], reward=0.25)

        self.population.append(c2)
        self.assertEqual(0.125, self.population.get_maximum_fitness())

        # C3 - does anticipate change and is quite good
        c3 = Classifier(effect=['1', '#', '#', '#', '#', '#', '#', '#'], quality=0.8, reward=5)

        self.population.append(c3)
        self.assertEqual(4, self.population.get_maximum_fitness())

    def test_should_return_best_fitness_action(self):
        # C1 - does not anticipate change
        c1 = Classifier(action=1)
        self.population.append(c1)

        # Some random action should be selected here
        best_action = self.population.choose_best_fitness_action()
        self.assertIsNotNone(best_action)

        # C2 - does anticipate some change
        c2 = Classifier(action=2, effect=['1', '#', '#', '#', '0', '#', '#', '#'], reward=0.25)
        self.population.append(c2)

        # Here C2 action should be selected
        best_action = self.population.choose_best_fitness_action()
        self.assertEqual(2, best_action)

        # C3 - does anticipate change and is quite good
        c3 = Classifier(action=3, effect=['1', '#', '#', '#', '#', '#', '#', '#'], quality=0.8, reward=5)
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
        self.assertEqual(1, self.population.choose_action_from_knowledge_array())

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
        self.assertEqual(7, self.population.choose_action_from_knowledge_array())

    def test_should_get_similar_classifier(self):
        self.population.append(Classifier(action=1))
        self.population.append(Classifier(action=2))
        self.population.append(Classifier(action=3))

        # No similar classifiers exist
        self.assertIsNone(self.population.get_similar(Classifier(action=4)))

        # Should find similar classifier
        self.assertIsNotNone(self.population.get_similar(Classifier(action=2)))

    def test_should_add_matching_classifiers(self):
        cls_lst = ClassifiersList()
        cls_lst.append(Classifier(
            condition=['#', '0', '#', '#', '#', '#', '#', '#']
        ))
        cls_lst.append(Classifier(
            condition=['1', '#', '#', '#', '#', '#', '#', '#']
        ))
        cls_lst.append(Classifier(
            condition=['0', '1', '#', '#', '#', '#', '#', '#']
        ))

        # Based on the perception should add 2 classifiers to the population
        situation = Perception(['1', '0', '1', '1', '0', '0', '0', '0'])
        self.population.add_matching_classifiers(cls_lst, situation)
        self.assertEqual(2, len(self.population))

        # Try to add last one
        situation = Perception(['0', '1', '1', '1', '0', '0', '0', '0'])
        self.population.add_matching_classifiers(cls_lst, situation)
        self.assertEqual(3, len(self.population))

    def test_should_apply_reinforcement_learning(self):
        c1 = Classifier()
        c1.r = 34.29
        c1.ir = 11.29
        self.population.append(c1)

        self.population.apply_reinforcement_learning(0, 28.79)

        self.assertAlmostEqual(33.94, self.population[0].r, 2)
        self.assertAlmostEqual(10.73, self.population[0].ir, 2)

