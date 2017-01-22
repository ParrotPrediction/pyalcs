import unittest

from alcs.agent.acs2 import Classifier
from alcs.agent.acs2.ALP import _does_anticipate_correctly


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
