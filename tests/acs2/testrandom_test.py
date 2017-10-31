import unittest
from alcs.acs2.testrandom import TestRandom, TestSample


class TestRandomTest(unittest.TestCase):

    def test_overall_numerosity(self):
        pass

    def test_testrandom(self):
        f = TestRandom([0.1, 0.2, 0.3])
        self.assertEqual(0.1, f())
        self.assertEqual(0.2, f())
        self.assertEqual(0.3, f())

    def test_testsample1(self):
        sample_func = TestSample([2, 0, 1])
        self.assertEqual([15, 3, 14], sample_func([3, 14, 15], 3))

    def test_testsample2(self):
        sample_func = TestSample([2, 0, 1])
        self.assertEqual([15, 3, 14], sample_func([3, 14, 15, 92, 6], 3))

    def test_testsample3(self):
        sample_func = TestSample([1, 15, 2, 15])
        self.assertEqual([14, 3, 15], sample_func([3, 14, 15], 3))

    def test_testsample4(self):
        sample_func = TestSample([10, 2, 1, 15])
        self.assertEqual([3, 15, 14], sample_func([3, 14, 15, 92, 6], 3))
