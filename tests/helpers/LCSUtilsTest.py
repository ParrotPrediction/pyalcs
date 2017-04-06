import unittest

from alcs.agent.acs2 import Classifier
from alcs.helpers.lcs_utils import get_dynamic_exploration_probability,\
    unwind_micro_classifiers


class LCSUtilsTest(unittest.TestCase):

    def test_shouldGenerateDynamicExplorationRate(self):
        max_steps = 500

        self.assertAlmostEqual(
            0.998,
            get_dynamic_exploration_probability(1, max_steps),
            places=3)

        self.assertAlmostEqual(
            0.5,
            get_dynamic_exploration_probability(250, max_steps),
            places=3)

        self.assertAlmostEqual(
            0.04,
            get_dynamic_exploration_probability(480, max_steps),
            places=3)

    def test_should_unwind_classifiers(self):
        macro_classifiers = [
            self._create_classifier(1),
            self._create_classifier(3)
        ]

        micro_classifiers = unwind_micro_classifiers(macro_classifiers)

        self.assertEqual(4, len(micro_classifiers))

    @staticmethod
    def _create_classifier(num: int):
        cl = Classifier()

        cl.num = num

        return cl
