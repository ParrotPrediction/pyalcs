import unittest
from agent import Classifier


class ClassifierTest(unittest.TestCase):

    def setUp(self):
        self.baseClassifier = Classifier()
        self.baseClassifier.condition = ['#', '1', '#', '#']
        self.baseClassifier.effect = ['#', '#', '1', '1']
        self.baseClassifier.action = 1

    def test_classifiers_should_be_equal(self):
        c2 = Classifier()
        c2.condition = ['#', '1', '#', '#']
        c2.effect = ['#', '#', '1', '1']
        c2.action = 1

        self.assertTrue(self.baseClassifier == c2)

    def test_classifiers_with_different_conditions_should_not_be_equal(self):
        c2 = Classifier()
        c2.condition = ['#', '1', '1', '#']
        c2.action = 1

        self.assertTrue(self.baseClassifier != c2)

    def test_classifiers_with_different_actions_should_not_be_equal(self):
        c2 = Classifier()
        c2.condition = ['#', '1', '#', '#']
        c2.action = 2

        self.assertTrue(self.baseClassifier != c2)

    def test_should_subsume_another_more_general_classifier(self):
        c2 = Classifier()
        c2.condition = ['#', '#', '#', '#']  # more general condition part
        c2.effect = ['#', '#', '1', '1']  # the same effect part

        self.assertTrue(self.baseClassifier.is_subsumer(c2, -1, -1))

    def test_should_not_subsume_another_less_general_classifier(self):
        c2 = Classifier()
        c2.condition = ['#', '1', '1', '#']  # less general part
        c2.effect = ['#', '#', '1', '1']  # the same effect part

        self.assertFalse(self.baseClassifier.is_subsumer(c2, -1, -1))

    def test_should_not_subsume_another_classifier_with_different_effect(self):
        c2 = Classifier()
        c2.condition = ['#', '1', '#', '#']
        c2.effect = ['#', '#', '#', '1']

        self.assertFalse(self.baseClassifier.is_subsumer(c2, -1, -1))

    def test_base_classifier_should_be_more_general(self):
        c2 = Classifier()
        c2.condition = ['1', '1', '#', '#']

        self.assertTrue(self.baseClassifier.is_more_general(c2))

    def test_base_classifier_should_be_less_general(self):
        c2 = Classifier()
        c2.condition = ['#', '#', '#', '#']

        self.assertFalse(self.baseClassifier.is_more_general(c2))

    def test_should_calculate_fitness(self):
        c2 = Classifier()
        c2.q = 0.5
        c2.r = 0.5

        self.assertEqual(0.25, c2.fitness())
