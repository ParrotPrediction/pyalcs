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