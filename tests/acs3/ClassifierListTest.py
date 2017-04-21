import unittest

from alcs.agent.Perception import Perception
from alcs.agent.acs3 import ClassifiersList, Classifier, Condition, Effect


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
