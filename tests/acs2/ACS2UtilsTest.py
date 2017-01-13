import unittest
from agent import ACS2Utils
from agent.acs2.Classifier import Classifier
from agent.acs2 import Constants as c


class ACS2UtilsTest(unittest.TestCase):

    def setUp(self):
        self.clsf1 = Classifier()
        self.clsf1.condition = ['#', '2', '#', '#', '1']
        self.clsf1.action = 1

        self.clsf2 = Classifier()
        self.clsf2.condition = ['#', '#', '1', '#', '1']
        self.clsf2.action = 2

        self.clsf3 = Classifier()
        self.clsf3.condition = ['1', '#', '#', '1', '2']
        self.clsf3.action = 2

        self.classifiers = [self.clsf1, self.clsf2, self.clsf3]

    def test_should_generate_initial_classifier_set(self):
        n_cls = 4  # Number of classifiers
        general_perception = []
        for i in range(n_cls):
            general_perception.append(c.CLASSIFIER_WILDCARD)

        initial_classifiers = ACS2Utils.generate_initial_classifiers(n_cls)
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
            ACS2Utils.generate_match_set(
                self.classifiers, ['3', '2', '1', '3', '1'])
        )

        self.assertListEqual(
            [],
            ACS2Utils.generate_match_set(
                self.classifiers, ['1', '2', '1', '3', '3']
            )
        )

        self.assertListEqual(
            [self.clsf3],
            ACS2Utils.generate_match_set(
                self.classifiers, ['1', '2', '1', '1', '2']
            )
        )

    def test_should_generate_action_set(self):
        self.assertListEqual(
            [],
            ACS2Utils.generate_action_set(self.classifiers, 3))

        self.assertListEqual(
            [self.clsf2, self.clsf3],
            ACS2Utils.generate_action_set(self.classifiers, 2)
        )

        self.assertListEqual(
            [self.clsf1],
            ACS2Utils.generate_action_set(self.classifiers, 1)
        )

    @unittest.skip("TODO")
    def test_should_choose_action(self):
        pass

    def test_classifer_should_match_perception(self):
        self.assertTrue(ACS2Utils._does_match(self.clsf1,
                                              ['1', '2', '-1', '1', '1']))

        # second element different
        self.assertFalse(ACS2Utils._does_match(self.clsf1,
                                               ['0', '1', '1', '1', '1']))

        self.assertRaises(ValueError,
                          ACS2Utils._does_match,
                          self.clsf1,
                          ['0', '1'])
