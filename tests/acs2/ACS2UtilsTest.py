import unittest

from alcs.agent.acs2.ACS2Utils import *
from alcs.agent.acs2.ACS2Utils import _does_match


class ACS2UtilsTest(unittest.TestCase):

    def setUp(self):
        self.clsf1 = __class__._create_classifier(['#', '2', '#', '#', '1'],
                                                  1, 0.5, 0.1)

        self.clsf2 = __class__._create_classifier(['#', '#', '1', '#', '1'],
                                                  2, 0.3, 0.2)

        self.clsf3 = __class__._create_classifier(['1', '#', '#', '1', '2'],
                                                  2, 0.8, 3)

        self.classifiers = [self.clsf1, self.clsf2, self.clsf3]

    def test_should_generate_initial_classifier_set(self):
        n_cls = 4  # Number of classifiers
        general_perception = []
        for i in range(n_cls):
            general_perception.append(c.CLASSIFIER_WILDCARD)

        initial_classifiers = generate_initial_classifiers(n_cls)
        self.assertEqual(n_cls, len(initial_classifiers))

        for i in range(n_cls):
            cls = initial_classifiers[i]
            self.assertListEqual(general_perception, cls.condition)
            self.assertListEqual(general_perception, cls.effect)
            self.assertEqual(i, cls.action)
            self.assertEqual(0.5, cls.q)
            self.assertEqual(0, cls.r)

    def test_should_generate_match_set(self):
        self.assertListEqual(
            [self.clsf1, self.clsf2],
            generate_match_set(self.classifiers, ['3', '2', '1', '3', '1'])
        )

        self.assertListEqual(
            [],
            generate_match_set(self.classifiers, ['1', '2', '1', '3', '3'])
        )

        self.assertListEqual(
            [self.clsf3],
            generate_match_set(self.classifiers, ['1', '2', '1', '1', '2'])
        )

    def test_should_generate_action_set(self):
        self.assertListEqual(
            [],
            generate_action_set(self.classifiers, 3)
        )

        self.assertListEqual(
            [self.clsf2, self.clsf3],
            generate_action_set(self.classifiers, 2)
        )

        self.assertListEqual(
            [self.clsf1],
            generate_action_set(self.classifiers, 1)
        )

    def test_should_choose_action(self):
        # all classifiers have general E parts
        # with epsilon = 0 we should always return action
        # from the first classifier (not randomly)
        self.assertEquals(1, choose_action(self.classifiers, epsilon=0))

        # now lets change the effect parts
        # in this case action from classifier with the best
        # fitness score is selected (not randomly)
        self.clsf1.effect = ['1', '2', '#', '1']
        self.clsf2.effect = ['#', '2', '#', '1']
        self.clsf3.effect = ['#', '#', '2', '1']

        self.assertEquals(2, choose_action(self.classifiers, epsilon=0))

        # in the last case we can see if we repeat the experiment
        # multiple times selecting randomly actions the proportions will
        # be correct (see number of possible actions in constants).
        trials = 10000
        ideal_counts = trials / c.NUMBER_OF_POSSIBLE_ACTIONS
        delta = trials / 100

        random_actions = [choose_action(self.classifiers, 1)
                          for _ in range(trials)]

        for action in range(c.NUMBER_OF_POSSIBLE_ACTIONS):
            real_counts = sum(1 for x in random_actions if x == action)
            self.assertAlmostEqual(ideal_counts, real_counts, delta=delta)

    def test_classifer_should_match_perception(self):
        self.assertTrue(_does_match(self.clsf1, ['1', '2', '-1', '1', '1']))

        # second element different
        self.assertFalse(_does_match(self.clsf1, ['0', '1', '1', '1', '1']))

        self.assertRaises(ValueError, _does_match, self.clsf1, ['0', '1'])

    @staticmethod
    def _create_classifier(condition: list,
                           action: int,
                           q: float,
                           r: float) -> Classifier:
        cl = Classifier()
        cl.condition = condition
        cl.action = action
        cl.q = q
        cl.r = r

        return cl
