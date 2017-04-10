import unittest

from alcs.agent.acs2 import Classifier
from alcs.environment.maze import Maze
from alcs.helpers.metrics import ActualStep, \
    ClassifierPopulationSize, \
    AchievedKnowledge


class MetricsTest(unittest.TestCase):

    def test_should_fail_on_missing_parameter(self):
        metric_handler = ActualStep("actual_step")

        self.assertIsNotNone(metric_handler.get(step=50))
        self.assertRaises(ValueError, metric_handler.get, unknown_param=50)

    def test_should_record_actual_step(self):
        metric_handler = ActualStep("actual_step")
        res = metric_handler.get(step=50)

        self.assertEqual(('actual_step', 50), res)

    def test_should_count_classifier_population(self):
        cl1 = Classifier()
        cl2 = Classifier()

        classifiers = [cl1, cl2]

        metric_handler = ClassifierPopulationSize("classifier_population_size")
        res = metric_handler.get(classifiers=classifiers)

        self.assertEqual(("classifier_population_size", 2), res)

    def test_should_calculate_knowledge(self):
        maze = Maze('tests/maze/test1.maze')

        metric_handler = AchievedKnowledge("achieved_knowledge")
        res = metric_handler.get(classifiers=[], maze=maze)
