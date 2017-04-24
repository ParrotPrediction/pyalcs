import unittest

from alcs.agent.Perception import Perception
from alcs.agent.acs3 import ClassifiersList, Classifier, Condition, Action, Effect

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
        c2 = Classifier()
        c2.condition = Condition(['1', '#', '#', '#', '0', '#', '#', '#'])

        # C3 - non-matching condition
        c3 = Classifier()
        c3.condition = Condition(['0', '#', '#', '#', '1', '#', '#', '#'])

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

        action_set = ClassifiersList.form_action_set(self.population, Action(0))
        self.assertEqual(2, len(action_set))
        self.assertIn(c0, action_set)
        self.assertIn(c01, action_set)

        action_set = ClassifiersList.form_action_set(self.population, Action(1))
        self.assertEqual(1, len(action_set))
        self.assertIn(c1, action_set)

    def test_should_calculate_maximum_fitness(self):
        # C1 - does not anticipate change
        c1 = Classifier()

        self.population.append(c1)
        self.assertEqual(0.0, self.population.get_maximum_fitness())

        # C2 - does anticipate some change
        c2 = Classifier()
        c2.effect = Effect(['1', '#', '#', '#', '0', '#', '#', '#'])
        c2.r = 0.25

        self.population.append(c2)
        self.assertEqual(0.125, self.population.get_maximum_fitness())

        # C3 - does anticipate change and is quite good
        c3 = Classifier()
        c3.effect = Effect(['1', '#', '#', '#', '#', '#', '#', '#'])
        c3.q = 0.8
        c3.r = 5

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
        c2 = Classifier(action=2)
        c2.effect = Effect(['1', '#', '#', '#', '0', '#', '#', '#'])
        c2.r = 0.25

        self.population.append(c2)

        # Here C2 action should be selected
        best_action = self.population.choose_best_fitness_action()
        self.assertEqual(Action(2), best_action)

        # C3 - does anticipate change and is quite good
        c3 = Classifier(action=3)
        c3.effect = Effect(['1', '#', '#', '#', '#', '#', '#', '#'])
        c3.q = 0.8
        c3.r = 5

        self.population.append(c3)

        # Here C3 has the biggest fitness score
        best_action = self.population.choose_best_fitness_action()
        self.assertEqual(Action(3), best_action)

    def test_should_return_random_action(self):
        random_actions = []

        for _ in range(0, 500):
            random_actions.append(self.population.choose_random_action())

        min_action = min(random_actions, key=lambda a: a.action)
        max_action = max(random_actions, key=lambda a: a.action)

        self.assertEqual(Action(0), min_action)
        self.assertEqual(Action(7), max_action)

    def test_should_return_latest_action(self):
        c0 = Classifier(action=0)
        c0.talp = 1

        self.population.append(c0)

        # Should return first action with no classifiers
        self.assertEqual(Action(1), self.population.choose_latest_action())

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
        self.assertEqual(Action(7), self.population.choose_latest_action())

    def test_should_return_worst_quality_action(self):
        c0 = Classifier(action=0)
        self.population.append(c0)

        # Should return C1 (because it's first not mentioned)
        self.assertEqual(Action(1), self.population.choose_action_from_knowledge_array())

        # Add rest of classifiers
        c1 = Classifier(action=1)
        c1.num = 31
        c1.q = 0.72
        self.population.append(c1)

        c2 = Classifier(action=2)
        c2.num = 2
        c2.q = 0.6
        self.population.append(c2)

        c3 = Classifier(action=3)
        c3.num = 2
        c3.q = 0.63
        self.population.append(c3)

        c4 = Classifier(action=4)
        c4.num = 7
        c4.q = 0.75
        self.population.append(c4)

        c5 = Classifier(action=5)
        c5.num = 1
        c5.q = 0.63
        self.population.append(c5)

        c6 = Classifier(action=6)
        c6.num = 6
        c6.q = 0.52
        self.population.append(c6)

        c7 = Classifier(action=7)
        c7.num = 10
        c7.q = 0.36
        self.population.append(c7)

        # Classifier C7 should be the worst here
        self.assertEqual(Action(7), self.population.choose_action_from_knowledge_array())
