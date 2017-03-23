import unittest

from alcs.helpers.lcs_utils import get_dynamic_exploration_probability


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
