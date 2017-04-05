import unittest

from alcs.agent.acs2 import Classifier


class ClassifierTest(unittest.TestCase):

    def setUp(self):
        self.baseCls = Classifier()
        self.baseCls.condition = ['#', '1', '#', '#', '#', '#', '#', '#']
        self.baseCls.effect = ['#', '#', '1', '1', '#', '#', '#', '#']
        self.baseCls.action = 1

    def test_classifiers_should_be_equal(self):
        c2 = Classifier()
        c2.condition = ['#', '1', '#', '#', '#', '#', '#', '#']
        c2.effect = ['#', '#', '1', '1', '#', '#', '#', '#']
        c2.action = 1

        self.assertTrue(self.baseCls == c2)

    def test_classifiers_with_different_conditions_should_not_be_equal(self):
        c2 = Classifier()
        c2.condition = ['#', '1', '1', '#']
        c2.action = 1

        self.assertTrue(self.baseCls != c2)

    def test_classifiers_with_different_actions_should_not_be_equal(self):
        c2 = Classifier()
        c2.condition = ['#', '1', '#', '#']
        c2.action = 2

        self.assertTrue(self.baseCls != c2)

    def test_should_subsume_another_more_general_classifier(self):
        c2 = Classifier()
        # more general condition part
        c2.condition = ['#', '#', '#', '#', '#', '#', '#', '#']
        # the same effect part
        c2.effect = ['#', '#', '1', '1', '#', '#', '#', '#']

        self.assertTrue(self.baseCls.can_subsume(c2, -1, -1))

    def test_should_not_subsume_another_less_general_classifier(self):
        c2 = Classifier()
        # less general part
        c2.condition = ['#', '1', '1', '#', '#', '#', '#', '#']
        # the same effect part
        c2.effect = ['#', '#', '1', '1', '#', '#', '#', '#']

        self.assertFalse(self.baseCls.can_subsume(c2, -1, -1))

    def test_should_not_subsume_another_classifier_with_different_effect(self):
        c2 = Classifier()
        c2.condition = ['#', '1', '#', '#', '#', '#', '#', '#']
        c2.effect = ['#', '#', '#', '1', '#', '#', '#', '#']

        self.assertFalse(self.baseCls.can_subsume(c2, -1, -1))

    def test_base_classifier_should_be_more_general(self):
        c2 = Classifier()
        c2.condition = ['1', '1', '#', '#', '#', '#', '#', '#']

        self.assertTrue(self.baseCls.is_more_general(c2))

    def test_base_classifier_should_be_less_general(self):
        c2 = Classifier()
        c2.condition = ['#', '#', '#', '#']

        self.assertFalse(self.baseCls.is_more_general(c2))

    def test_should_calculate_fitness(self):
        c2 = Classifier()
        c2.q = 0.5
        c2.r = 0.5

        self.assertEqual(0.25, c2.fitness())

    def test_should_copy_classifier(self):
        copied = Classifier.copy_from(self.baseCls)

        self.assertListEqual(self.baseCls.condition, copied.condition)
        self.assertEqual(self.baseCls.action, copied.action)
        self.assertListEqual(self.baseCls.effect, copied.effect)
        self.assertEqual(self.baseCls.mark, copied.mark)
        self.assertEqual(self.baseCls.q, copied.q)
        self.assertEqual(self.baseCls.r, copied.r)
        self.assertEqual(self.baseCls.ir, copied.ir)
        self.assertEqual(self.baseCls.t, copied.t)
        self.assertEqual(self.baseCls.t_ga, copied.t_ga)
        self.assertEqual(self.baseCls.t_alp, copied.t_alp)
        self.assertEqual(self.baseCls.aav, copied.aav)
        self.assertEqual(self.baseCls.exp, copied.exp)
        self.assertEqual(self.baseCls.num, copied.num)

        # Check if a new reference for object is created
        self.assertFalse(copied.condition is self.baseCls.condition)

    def test_should_set_mark_on_classifier(self):
        previous_perception = ['1', '1', '2', '1']
        perception = ['1', '2', '1', '1']

        self.baseCls.set_mark(previous_perception, perception)

        self.assertTrue(self.baseCls.mark[0] == set())
        self.assertTrue(self.baseCls.mark[1] == {'2'})
        self.assertTrue(self.baseCls.mark[2] == {'1'})
        self.assertTrue(self.baseCls.mark[3] == {'1'})

    def test_should_identify_marked_classifier(self):
        self.assertFalse(Classifier.is_marked(self.baseCls.mark))

        previous_perception = ['1', '1', '2', '1']
        perception = ['1', '1', '2', '1']
        self.baseCls.set_mark(previous_perception, perception)

        self.assertTrue(Classifier.is_marked(self.baseCls.mark))

    def test_should_calculate_specificity_measure(self):
        cl = Classifier()

        cl.condition = ['1', '2', '1', '0', '1', '2', '1', '4']
        self.assertEqual(1, cl.get_condition_specificity())

        cl.condition = ['#', '#', '#', '#', '#', '#', '#', '#']
        self.assertEqual(0, cl.get_condition_specificity())

        cl.condition = ['1', '#', '1', '#', '#', '1', '#', '4']
        self.assertEqual(0.5, cl.get_condition_specificity())
