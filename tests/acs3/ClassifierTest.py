import unittest

from alcs.agent import Perception
from alcs.agent.acs3 import Classifier, Condition, Action, Effect


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

    def test_should_anticipate_correctly(self):
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '1', '0', '0', '1', '1', '0', '1'])

        self.cls.effect = Effect(['#', '1', '#', '#', '#', '#', '0', '#'])

        self.assertTrue(self.cls.does_anticipate_correctly(p0, p1))

    def test_should_update_reward(self):
        self.cls.update_reward(1000)
        self.assertEqual(50.0, self.cls.r)

    def test_should_update_intermediate_reward(self):
        self.cls.update_intermediate_reward(1000)
        self.assertEqual(50.0, self.cls.ir)

    def test_should_increase_quality(self):
        self.cls.increase_quality()
        self.assertEqual(0.525, self.cls.q)

    def test_should_cover_triple(self):
        action_no = 2
        time = 123
        p0 = Perception(['0', '1', '0', '0', '1', '1', '0', '1'])
        p1 = Perception(['0', '0', '0', '1', '1', '1', '1', '1'])

        new_cl = Classifier.cover_triple(p0, action_no, p1, time)

        self.assertEqual(Condition(['#', '1', '#', '0', '#', '#', '0', '#']), new_cl.condition)
        self.assertEqual(Action(2), new_cl.action)
        self.assertEqual(Effect(['#', '0', '#', '1', '#', '#', '1', '#']), new_cl.effect)
        self.assertEqual(0.5, new_cl.q)
        self.assertEqual(0.5, new_cl.r)
        self.assertEqual(0, new_cl.ir)
        self.assertEqual(0, new_cl.tav)
        self.assertEqual(time, new_cl.tga)
        self.assertEqual(time, new_cl.talp)
        self.assertEqual(1, new_cl.num)
        self.assertEqual(1, new_cl.exp)

    def test_should_count_specified_unchanging_attributes(self):
        cl1 = Classifier(
            condition=Condition(['#', '#', '#', '#', '#', '#', '0', '#']),
            effect=Effect(      ['#', '#', '#', '#', '#', '#', '#', '#'])
        )
        self.assertEqual(1, cl1.specified_unchanging_attributes)

        cl2 = Classifier(
            condition=Condition(['#', '#', '#', '#', '#', '0', '#', '0']),
            effect=Effect(      ['#', '#', '#', '#', '#', '#', '#', '#'])
        )
        self.assertEqual(2, cl2.specified_unchanging_attributes)

        cl3 = Classifier(
            condition=Condition(['1', '0', '0', '0', '0', '0', '0', '1']),
            effect=Effect(      ['#', '#', '#', '#', '1', '#', '1', '#'])
        )
        self.assertEqual(6, cl3.specified_unchanging_attributes)

        cl4 = Classifier(
            condition=Condition(['1', '#', '0', '#', '1', '0', '1', '1']),
            effect=Effect(      ['0', '#', '#', '#', '#', '1', '#', '#'])
        )
        self.assertEqual(4, cl4.specified_unchanging_attributes)

        cl5 = Classifier(
            condition=Condition(['1', '#', '#', '#', '1', '0', '1', '1']),
            effect=Effect(      ['0', '#', '#', '#', '#', '1', '#', '#'])
        )
        self.assertEqual(3, cl5.specified_unchanging_attributes)

    def test_should_copy_classifier(self):
        # TODO: NYI
        pass
