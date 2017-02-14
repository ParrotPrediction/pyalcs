import unittest

from alcs.agent.acs2 import Classifier
from alcs.agent.acs2.ALP import _does_anticipate_correctly,\
    _cover_triple, \
    _unexpected_case


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

    def test_should_handle_unexpected_case_no_specialization_possible(self):
        cl = Classifier()
        # We are looking only at non-pass-through elements

        # First element in effect part does not anticipate correctly.
        # We cannot specialize it (already specialized).
        cl.effect = ['1', '#', '0', '0']
        perception = ['0', '0', '0', '0']  # wrong anticipation
        previous_perception = ['0', '0', '1', '1']
        self.assertIsNone(_unexpected_case(cl,
                                           perception,
                                           previous_perception))

        # Counter example - previous element in effect part does anticipate
        # correctly. Here a new classifier can be created.
        cl.effect = ['1', '#', '0', '0']
        perception = ['1', '0', '0', '0']  # correct anticipation
        previous_perception = ['0', '0', '1', '1']
        self.assertIsNotNone(_unexpected_case(cl,
                                              perception,
                                              previous_perception))

        # If there is specialized element in effect part, but previous
        # and current perception are the same - it's an error. There
        # should be a pass-through in effect there.
        cl.effect = ['1', '#', '0', '0']  # Last element should be '#'
        perception = ['1', '1', '0', '0']
        previous_perception = ['0', '1', '1', '0']
        self.assertIsNone(_unexpected_case(cl,
                                           perception,
                                           previous_perception))

        # Counter example
        cl.effect = ['1', '#', '0', '#']  # Last element is '#'
        perception = ['1', '1', '0', '0']
        previous_perception = ['0', '1', '1', '0']
        self.assertIsNotNone(_unexpected_case(cl,
                                              perception,
                                              previous_perception))

    def test_should_handle_unexpected_case(self):
        # Here we try to 'specialize' the classifier by investigating
        # it's pass-through symbols
        cl = Classifier()

        # There is a pass-through but previous and current perception
        # are different
        cl.condition = ['1', '0', '0', '1']
        cl.effect = ['#', '#', '#', '#']
        previous_perception = ['0', '1', '0', '1']
        perception = ['1', '0', '0', '1']

        new_cl = _unexpected_case(cl, perception, previous_perception)

        # Make sure that condition and effect parts of the new
        # classifier were changed on pass-through elements.
        self.assertEqual(['0', '1', '0', '1'], new_cl.condition)
        self.assertEqual(['1', '0', '#', '#'], new_cl.effect)
        self.assertEqual(1, new_cl.exp)

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
