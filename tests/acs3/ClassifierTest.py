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

    def test_should_generate_new_classifier_from_unexpected_case(self):
        self.cls = Classifier(action=2)

        p0 = Perception(['0', '1', '1', '0', '0', '0', '0', '0'])
        p1 = Perception(['1', '0', '1', '0', '0', '0', '1', '0'])
        time = 14

        new_cls = self.cls.unexpected_case(p0, p1, time)

        # Quality should be decreased
        self.assertEqual(0.475, self.cls.q)

        # Should be marked with previous perception
        for mark_attrib in self.cls.mark:
            self.assertEqual(1, len(mark_attrib))

        self.assertIn('0', self.cls.mark[0])
        self.assertIn('1', self.cls.mark[1])
        self.assertIn('1', self.cls.mark[2])
        self.assertIn('0', self.cls.mark[3])
        self.assertIn('0', self.cls.mark[4])
        self.assertIn('0', self.cls.mark[5])
        self.assertIn('0', self.cls.mark[6])
        self.assertIn('0', self.cls.mark[7])

        # New classifier should not be the same object
        self.assertFalse(self.cls is new_cls)

        # Check attributes of a new classifier
        self.assertEqual(
            Condition(['0', '1', '#', '#', '#', '#', '0', '#']),
            new_cls.condition
        )
        self.assertEqual(Action(2), new_cls.action)
        self.assertEqual(
            Effect(['1', '0', '#', '#', '#', '#', '1', '#']),
            new_cls.effect
        )

        # There should be no mark
        for mark_attrib in new_cls.mark:
            self.assertEqual(0, len(mark_attrib))

        self.assertEqual(0.5, new_cls.q)
        self.assertEqual(self.cls.r, new_cls.r)
        self.assertEqual(time, new_cls.tga)
        self.assertEqual(time, new_cls.talp)

    def test_should_not_generate_new_classifier_from_unexpected_case(self):
        self.cls = Classifier(
            condition=['#', '#', '#', '#', '1', '#', '0', '#'],
            action=5,
            effect=Effect(['#', '#', '#', '#', '0', '#', '1', '#']),
            quality=0.475
        )

        self.cls.mark[0] = '1'
        self.cls.mark[1] = '1'
        self.cls.mark[2] = '0'
        self.cls.mark[3] = '1'
        self.cls.mark[5] = '1'
        self.cls.mark[7] = '1'

        p0 = Perception(['1', '1', '0', '1', '1', '1', '0', '1'])
        p1 = Perception(['0', '1', '1', '0', '0', '0', '0', '0'])
        time = 20

        new_cls = self.cls.unexpected_case(p0, p1, time)

        # Quality should be decreased
        self.assertEqual(0.45125, self.cls.q)

        # No classifier should be generated here
        self.assertIsNone(new_cls)

    def test_should_copy_classifier(self):
        # TODO: NYI
        pass
