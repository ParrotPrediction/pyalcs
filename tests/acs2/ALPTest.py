import unittest

from alcs.agent.acs2 import Classifier
from alcs.agent.acs2.ALP import _does_anticipate_correctly,\
    _cover_triple


class ALPTest(unittest.TestCase):

    def test_should_anticipate_correctly(self):
        cl = Classifier()

        # In this case classifier should be ['#', '#', '#', '#],
        # because the perception values pass-through
        previous_perception = ['1', '2', '1', '1']
        perception = ['1', '2', '1', '1']
        cl.effect = ['1', '#', '1', '1']

        self.assertFalse(_does_anticipate_correctly(cl,
                                                    perception,
                                                    previous_perception))

        # In this case classifier wrongly predicts the effect of
        # the first input - '2' instead of '1'
        previous_perception = ['1', '2', '1', '1']
        perception = ['1', '2', '1', '1']
        cl.effect = ['2', '#', '#', '#']

        self.assertFalse(_does_anticipate_correctly(cl,
                                                    perception,
                                                    previous_perception))

        # In this case classifier says that all values are pass-through,
        # but that is not true
        previous_perception = ['1', '2', '2', '1']
        perception = ['1', '2', '1', '1']
        cl.effect = ['#', '#', '#']

        self.assertFalse(_does_anticipate_correctly(cl,
                                                    perception,
                                                    previous_perception))

        # In this case the classifier anticipates correctly
        previous_perception = ['1', '2', '1', '1']
        perception = ['1', '2', '2', '1']
        cl.effect = ['#', '#', '2', '#']

        self.assertTrue(_does_anticipate_correctly(cl,
                                                   perception,
                                                   previous_perception))

    def test_should_cover_triple(self):
        previous_perception = ['1', '2', '1', '1']
        perception = ['2', '2', '1', '1']
        action = 1
        time = 99

        new_cl = _cover_triple(previous_perception, perception, action, time)

        self.assertEqual(['1', '#', '#', '#'], new_cl.condition)
        self.assertEqual(['2', '#', '#', '#'], new_cl.effect)
        self.assertEqual(action, new_cl.action)
        self.assertEqual(0, new_cl.exp)
        self.assertEqual(0, new_cl.r)
        self.assertEqual(0, new_cl.aav)
        self.assertEqual(time, new_cl.t_alp)
        self.assertEqual(time, new_cl.t_ga)
        self.assertEqual(time, new_cl.t)
        self.assertEqual(1, new_cl.num)
