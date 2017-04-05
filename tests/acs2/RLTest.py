import unittest

from alcs.agent.acs2.ACS2Utils import *
from alcs.agent.acs2.RL import _calculate_maximum_payoff, apply_rl


class RLTest(unittest.TestCase):

    def setUp(self):
        self.cl1 = __class__._create_classifier(
            ['1', '#', '1', '#', '#', '#', '#', '#'],
            q=0.6, r=1, ir=2
        )

        self.cl2 = __class__._create_classifier(
            ['1', '1', '1', '#', '#', '#', '#', '#'],
            q=0.3, r=0.5, ir=0.8
        )

        self.cl3 = __class__._create_classifier(
            ['#', '#', '#', '#', '#', '#', '#', '#'],
            q=1, r=10, ir=0.4
        )

    def tearDown(self):
        self.cl1 = None
        self.cl2 = None
        self.cl3 = None

    def test_should_calculate_maximum_payoff(self):
        match_set = [self.cl1, self.cl2, self.cl3]
        desired_max_p = self.cl1.fitness()  # classifier with best fitness

        self.assertEqual(desired_max_p, _calculate_maximum_payoff(match_set))

    def test_should_calculate_maximum_payoff_with_general_classifiers(self):
        match_set = [self.cl3]
        self.assertEqual(0, _calculate_maximum_payoff(match_set))

    def test_should_apply_rl_with_default_parameters(self):
        match_set = [self.cl1, self.cl2, self.cl3]  # max_p = 0.6
        action_set = [self.cl1, self.cl2]
        obtained_reward = 1

        # default parameters
        apply_rl(match_set,
                 action_set,
                 obtained_reward,
                 learning_rate=0.2,
                 discount_factor=0.95)

        self.assertAlmostEqual(self.cl1.r, 1.114, places=3)
        self.assertAlmostEqual(self.cl1.ir, 1.8, places=3)

        self.assertAlmostEqual(self.cl2.r, 0.714, places=3)
        self.assertAlmostEqual(self.cl2.ir, 0.84, places=3)

    def test_should_apply_rl_with_no_learning_rate(self):
        match_set = [self.cl1, self.cl2, self.cl3]  # max_p = 0.6
        action_set = [self.cl1, self.cl2]
        obtained_reward = 1

        # default parameters
        apply_rl(match_set,
                 action_set,
                 obtained_reward,
                 learning_rate=0,
                 discount_factor=0.95)

        # No changes
        self.assertAlmostEqual(self.cl1.r, 1, places=3)
        self.assertAlmostEqual(self.cl1.ir, 2, places=3)

        self.assertAlmostEqual(self.cl2.r, 0.5, places=3)
        self.assertAlmostEqual(self.cl2.ir, 0.8, places=3)

    @staticmethod
    def _create_classifier(effect: list,
                           q: float,
                           r: float,
                           ir: float) -> Classifier:
        cl = Classifier()
        cl.effect = effect
        cl.q = q
        cl.r = r
        cl.ir = ir

        return cl
