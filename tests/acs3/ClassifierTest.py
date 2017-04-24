import unittest

from alcs.agent.acs3 import Classifier


class ClassifierTest(unittest.TestCase):

    def setUp(self):
        self.cls = Classifier()

    def test_should_calculate_fitness(self):
        self.cls.r = 0.25

        self.assertEqual(0.125, self.cls.fitness)

    def test_should_anticipate_change(self):
        self.assertFalse(self.cls.does_anticipate_change())

        self.cls.effect[1] = '1'
        self.assertTrue(self.cls.does_anticipate_change())

    def test_should_update_reward(self):
        self.cls.update_reward(1000)
        self.assertEqual(50.0, self.cls.r)

    def test_should_update_intermediate_reward(self):
        self.cls.update_intermediate_reward(1000)
        self.assertEqual(50.0, self.cls.ir)