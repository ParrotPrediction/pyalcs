import unittest

from alcs.agent.acs3 import Action


class ActionTest(unittest.TestCase):

    def test_should_set_action(self):
        a = Action(5)
        self.assertEqual(5, a.action)
