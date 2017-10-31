import unittest
from alcs.acs2.testrandom import TestRandom

class ClassifierListTest(unittest.TestCase):

    def test_testrandom(self):
        f = TestRandom([0.1, 0.2, 0.3])
        self.assertEqual(0.1, f())
        self.assertEqual(0.2, f())
        self.assertEqual(0.3, f())
