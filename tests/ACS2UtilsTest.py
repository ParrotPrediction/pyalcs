import unittest
from agent import ACS2Utils
from agent.acs2.Classifier import Classifier
from agent.acs2 import Constants as c


class ACS2UtilsTest(unittest.TestCase):

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

    @unittest.skip("TODO")
    def test_should_generate_match_set(self):
        pass

    @unittest.skip("TODO")
    def test_should_generate_action_set(self):
        pass

    @unittest.skip("TODO")
    def test_should_choose_action(self):
        pass

    def test_classifer_should_match_perception(self):
        cls = Classifier()
        cls.condition = ['#', '2', '#', '#', '1']

        self.assertTrue(ACS2Utils._does_match(cls, ['1', '2', '-1', '1', '1']))

        # second element different
        self.assertFalse(ACS2Utils._does_match(cls, ['0', '1', '1', '1', '1']))

        self.assertRaises(ValueError, ACS2Utils._does_match, cls, ['0', '1'])
