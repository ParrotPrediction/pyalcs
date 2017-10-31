import unittest

from alcs.acs2 import Classifier, Condition, Effect, PMark, Constants
from alcs import Perception
from alcs.acs2.testrandom import TestRandom

s = Constants.MU*0.5  # less then MU
b = 1-(1-Constants.MU)*0.5  # more then MU


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        self.cls = Classifier()

    def test_equality(self):
        cl = Classifier(action=1, numerosity=2)
        self.assertEqual(Classifier(action=1, numerosity=2), cl)

    def test_is_equally_general(self):
        c1 = Classifier(Condition('1#######'))
        self.assertTrue(c1.is_equally_general(Classifier(Condition('1#######'))))
        self.assertTrue(c1.is_equally_general(Classifier(Condition('0#######'))))
        self.assertTrue(c1.is_equally_general(Classifier(Condition('#0######'))))
        self.assertFalse(c1.is_equally_general(Classifier(Condition('#01#####'))))
        self.assertFalse(c1.is_equally_general(Classifier(Condition('########'))))

    def test_mutate_1(self):
        cls = Classifier(Condition('##011###'))
        cls.mutate(randomfunc=TestRandom([s, b, b]))
        self.assertEqual(Condition('###11###'), cls.condition)

    def test_mutate_2(self):
        cls = Classifier(Condition('##011###'))
        cls.mutate(randomfunc=TestRandom([b, b, s]))
        self.assertEqual(Condition('##01####'), cls.condition)

    def test_should_calculate_fitness(self):
        self.cls.r = 0.25
        self.assertEqual(0.125, self.cls.fitness)

    def test_should_anticipate_change(self):
        self.assertFalse(self.cls.does_anticipate_change())

        self.cls.effect[1] = '1'
        self.assertTrue(self.cls.does_anticipate_change())

    def test_should_anticipate_correctly(self):
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '1', '0', '0', '1', '1', '0', '1'])

        self.cls.effect = Effect(['#', '1', '#', '#', '#', '#', '0', '#'])

        self.assertTrue(self.cls.does_anticipate_correctly(p0, p1))

    def test_should_calculate_specificity_1(self):
        self.assertEqual(0, self.cls.specificity)

    def test_should_calculate_specificity_2(self):
        self.cls.condition = Condition(
            ['#', '1', '#', '0', '1', '#', '0', '#'])
        self.assertEqual(0.5, self.cls.specificity)

    def test_should_calculate_specificity_3(self):
        self.cls.condition = Condition(
            ['1', '1', '1', '0', '1', '0', '0', '1'])
        self.assertEqual(1, self.cls.specificity)

    def test_should_be_considered_as_reliable_1(self):
        # Given
        self.cls.q = 0.89

        # Then
        self.assertFalse(self.cls.is_reliable())

    def test_should_be_considered_as_reliable_2(self):
        # Given
        self.cls.q = 0.91

        # Then
        self.assertTrue(self.cls.is_reliable())

    def test_should_be_considered_as_inadequate_1(self):
        # Given
        self.cls.q = 0.50

        # Then
        self.assertFalse(self.cls.is_inadequate())

    def test_should_be_considered_as_inadequate_2(self):
        # Given
        self.cls.q = 0.09

        # Then
        self.assertTrue(self.cls.is_inadequate())

    def test_should_update_reward(self):
        self.cls.update_reward(1000)
        self.assertEqual(50.475, self.cls.r)

    def test_should_update_intermediate_reward(self):
        self.cls.update_intermediate_reward(1000)
        self.assertEqual(50.0, self.cls.ir)

    def test_should_increase_experience(self):
        # Given
        self.cls.exp = 5

        # When
        self.cls.increase_experience()

        # Then
        self.assertEqual(6, self.cls.exp)

    def test_should_increase_quality(self):
        # Given
        self.cls.q = 0.5

        # When
        self.cls.increase_quality()

        # Then
        self.assertEqual(0.525, self.cls.q)

    def test_should_decrease_quality(self):
        # Given
        self.cls.q = 0.47

        # When
        self.cls.decrease_quality()

        # Then
        self.assertAlmostEqual(0.45, self.cls.q, 2)

    def test_should_cover_triple(self):
        action_no = 2
        time = 123
        p0 = Perception(['0', '1', '0', '0', '1', '1', '0', '1'])
        p1 = Perception(['0', '0', '0', '1', '1', '1', '1', '1'])

        new_cl = Classifier.cover_triple(p0, action_no, p1, time)

        self.assertEqual(Condition(['#', '1', '#', '0', '#', '#', '0', '#']),
                         new_cl.condition)
        self.assertEqual(2, new_cl.action)
        self.assertEqual(Effect(['#', '0', '#', '1', '#', '#', '1', '#']),
                         new_cl.effect)
        self.assertEqual(0.5, new_cl.q)
        self.assertEqual(0.5, new_cl.r)
        self.assertEqual(0, new_cl.ir)
        self.assertEqual(0, new_cl.tav)
        self.assertEqual(time, new_cl.tga)
        self.assertEqual(time, new_cl.talp)
        self.assertEqual(1, new_cl.num)
        self.assertEqual(1, new_cl.exp)

    def test_should_specialize_1(self):
        # Given
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])

        # When
        self.cls.specialize(p0, p1)

        # Then
        self.assertEqual(Condition(['#', '#', '#', '#', '#', '#', '#', '#']),
                         self.cls.condition)
        self.assertEqual(Effect(['#', '#', '#', '#', '#', '#', '#', '#']),
                         self.cls.effect)

    def test_should_specialize_2(self):
        # Given
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '0', '0', '1', '1', '1', '1', '1'])

        # When
        self.cls.specialize(p0, p1)

        # Then
        self.assertEqual(Condition(['#', '#', '#', '0', '#', '#', '#', '#']),
                         self.cls.condition)
        self.assertEqual(Effect(['#', '#', '#', '1', '#', '#', '#', '#']),
                         self.cls.effect)

    def test_should_specialize_3(self):
        # Given
        p0 = Perception(['0', '1', '1', '1', '0', '1', '1', '1'])
        p1 = Perception(['1', '0', '1', '0', '1', '0', '1', '0'])
        self.cls.condition = Condition(
            ['0', '1', '#', '#', '#', '#', '#', '1'])
        self.cls.effect = Effect(['1', '0', '#', '#', '#', '#', '#', '0'])

        # When
        self.cls.specialize(p0, p1)
        # diff = Condition(['#', '#', '#', '1', '0', '1', '#', '#'])

        # Then
        self.assertEqual(6, self.cls.condition.specificity)
        self.assertEqual(Condition(['0', '1', '#', '1', '0', '1', '#', '1']),
                         self.cls.condition)

        self.assertEqual(6, self.cls.effect.number_of_specified_elements)
        self.assertEqual(Effect(['1', '0', '#', '0', '1', '0', '#', '0']),
                         self.cls.effect)

    def test_should_count_specified_unchanging_attributes_1(self):
        cl1 = Classifier(
            condition=Condition(['#', '#', '#', '#', '#', '#', '0', '#']),
            effect=Effect(['#', '#', '#', '#', '#', '#', '#', '#'])
        )
        self.assertEqual(1, cl1.specified_unchanging_attributes)

    def test_should_count_specified_unchanging_attributes_2(self):
        cl2 = Classifier(
            condition=Condition(['#', '#', '#', '#', '#', '0', '#', '0']),
            effect=Effect(['#', '#', '#', '#', '#', '#', '#', '#'])
        )
        self.assertEqual(2, cl2.specified_unchanging_attributes)

    def test_should_count_specified_unchanging_attributes_3(self):
        cl3 = Classifier(
            condition=Condition(['1', '0', '0', '0', '0', '0', '0', '1']),
            effect=Effect(['#', '#', '#', '#', '1', '#', '1', '#'])
        )
        self.assertEqual(6, cl3.specified_unchanging_attributes)

    def test_should_count_specified_unchanging_attributes_4(self):
        cl4 = Classifier(
            condition=Condition(['1', '#', '0', '#', '1', '0', '1', '1']),
            effect=Effect(['0', '#', '#', '#', '#', '1', '#', '#'])
        )
        self.assertEqual(4, cl4.specified_unchanging_attributes)

    def test_should_count_specified_unchanging_attributes_5(self):
        cl5 = Classifier(
            condition=Condition(['1', '#', '#', '#', '1', '0', '1', '1']),
            effect=Effect(['0', '#', '#', '#', '#', '1', '#', '#'])
        )
        self.assertEqual(3, cl5.specified_unchanging_attributes)

    def test_should_handle_expected_case_1(self):
        # Given
        self.cls.condition = ['#', '#', '#', '#', '#', '#', '#', '0']
        self.cls.q = 0.525
        p0 = Perception(['1', '1', '1', '1', '1', '0', '1', '0'])
        time = 47

        # When
        new_cls = self.cls.expected_case(p0, time)

        # Then
        self.assertIsNone(new_cls)
        self.assertAlmostEqual(0.54, self.cls.q, places=1)

    def test_should_handle_expected_case_2(self):
        # Given
        self.cls.condition = ['#', '0', '#', '#', '#', '#', '#', '#']
        self.cls.q = 0.521
        p0 = Perception(['1', '0', '1', '0', '1', '0', '0', '1'])
        time = 59

        # When
        new_cls = self.cls.expected_case(p0, time)

        # Then
        self.assertIsNone(new_cls)
        self.assertAlmostEqual(0.54, self.cls.q, places=1)

    def test_should_handle_expected_case_3(self):
        # Given
        p0 = Perception(['0', '0', '1', '1', '0', '0', '0', '0'])
        time = 26
        self.cls.action = 5
        self.cls.q = 0.46
        self.cls.mark[0] = '0'
        self.cls.mark[1] = '1'
        self.cls.mark[2] = '0'
        self.cls.mark[3] = '1'
        self.cls.mark[4] = '0'
        self.cls.mark[5] = '1'
        self.cls.mark[6] = '1'
        self.cls.mark[7] = '1'

        # When
        new_cls = self.cls.expected_case(p0, time)

        # Then
        self.assertIsNotNone(new_cls)
        # One `random` attribute gets specified
        self.assertEqual(1, new_cls.condition.specificity)
        self.assertEqual(Effect(['#', '#', '#', '#', '#', '#', '#', '#']),
                         new_cls.effect)
        self.assertEqual(5, new_cls.action)
        self.assertTrue(new_cls.mark.is_empty())
        self.assertEqual(0.5, new_cls.q)

    def test_should_handle_expected_case_4(self):
        # Given
        p0 = Perception(['1', '1', '1', '0', '1', '1', '0', '1'])
        time = 703
        self.cls.condition = Condition(
            ['1', '#', '#', '0', '1', '#', '0', '#'])
        self.cls.action = 7
        self.cls.effect = Effect(['0', '#', '#', '1', '0', '#', '1', '#'])
        self.cls.q = 0.47
        self.cls.mark[1].update(['0', '2'])
        self.cls.mark[2].update(['1'])
        self.cls.mark[5].update(['0', '1'])
        self.cls.mark[7].update(['1'])

        # When
        new_cls = self.cls.expected_case(p0, time)

        # Then
        self.assertIsNotNone(new_cls)
        # One `random` attribute gets specified
        self.assertEqual(5, new_cls.condition.specificity)
        self.assertEqual(Effect(['0', '#', '#', '1', '0', '#', '1', '#']),
                         new_cls.effect)
        self.assertEqual(7, new_cls.action)
        self.assertTrue(new_cls.mark.is_empty())
        self.assertEqual(0.5, new_cls.q)

    def test_should_handle_unexpected_case_1(self):
        self.cls = Classifier(action=2)

        p0 = Perception(['0', '1', '1', '0', '0', '0', '0', '0'])
        p1 = Perception(['1', '0', '1', '0', '0', '0', '1', '0'])
        time = 14

        new_cls = self.cls.unexpected_case(p0, p1, time)

        # Quality should be decreased
        self.assertEqual(0.475, self.cls.q)

        # Should be marked with previous perception
        for mark_attrib in self.cls.mark:
            self.assertEqual(1, len(mark_attrib))

        self.assertIn('0', self.cls.mark[0])
        self.assertIn('1', self.cls.mark[1])
        self.assertIn('1', self.cls.mark[2])
        self.assertIn('0', self.cls.mark[3])
        self.assertIn('0', self.cls.mark[4])
        self.assertIn('0', self.cls.mark[5])
        self.assertIn('0', self.cls.mark[6])
        self.assertIn('0', self.cls.mark[7])

        # New classifier should not be the same object
        self.assertFalse(self.cls is new_cls)

        # Check attributes of a new classifier
        self.assertEqual(
            Condition(['0', '1', '#', '#', '#', '#', '0', '#']),
            new_cls.condition
        )
        self.assertEqual(2, new_cls.action)
        self.assertEqual(
            Effect(['1', '0', '#', '#', '#', '#', '1', '#']),
            new_cls.effect
        )

        # There should be no mark
        for mark_attrib in new_cls.mark:
            self.assertEqual(0, len(mark_attrib))

        self.assertEqual(0.5, new_cls.q)
        self.assertEqual(self.cls.r, new_cls.r)
        self.assertEqual(time, new_cls.tga)
        self.assertEqual(time, new_cls.talp)

    def test_should_handle_unexpected_case_2(self):
        # Given
        self.cls.condition = Condition(
            ['#', '#', '#', '#', '#', '#', '#', '0'])
        self.cls.action = 4
        self.cls.effect = Effect()
        self.cls.mark[0].update([0, 1])
        self.cls.mark[1].update([0, 1])
        self.cls.mark[2].update([0, 1])
        self.cls.mark[3].update([0, 1])
        self.cls.mark[4].update([1])
        self.cls.mark[5].update([0, 1])
        self.cls.mark[6].update([0, 1])
        self.cls.q = 0.4

        p0 = Perception(['1', '0', '1', '0', '1', '0', '1', '0'])
        p1 = Perception(['1', '1', '0', '1', '1', '1', '0', '1'])
        time = 94

        # When
        new_cl = self.cls.unexpected_case(p0, p1, time)

        # Then
        self.assertEqual(new_cl.condition,
                         Condition(['#', '0', '1', '0', '#', '0', '1', '0']))
        self.assertEqual(new_cl.effect,
                         Effect(['#', '1', '0', '1', '#', '1', '0', '1']))
        self.assertTrue(new_cl.mark.is_empty())
        self.assertEqual(time, new_cl.tga)
        self.assertEqual(time, new_cl.talp)
        self.assertAlmostEqual(self.cls.q, 0.38, 1)

    def test_should_handle_unexpected_case_3(self):
        self.cls = Classifier(
            condition=['#', '#', '#', '#', '1', '#', '0', '#'],
            effect=Effect(['#', '#', '#', '#', '0', '#', '1', '#']),
            quality=0.475
        )

        self.cls.mark[0] = '1'
        self.cls.mark[1] = '1'
        self.cls.mark[2] = '0'
        self.cls.mark[3] = '1'
        self.cls.mark[5] = '1'
        self.cls.mark[7] = '1'

        p0 = Perception(['1', '1', '0', '1', '1', '1', '0', '1'])
        p1 = Perception(['0', '1', '1', '0', '0', '0', '0', '0'])
        time = 20

        new_cls = self.cls.unexpected_case(p0, p1, time)

        # Quality should be decreased
        self.assertEqual(0.45125, self.cls.q)

        # No classifier should be generated here
        self.assertIsNone(new_cls)

    def test_should_handle_unexpected_case_4(self):
        # Given
        self.cls = Classifier(
            condition=['#', '#', '1', '1', '#', '1', '#', '#'],
            effect=Effect(['#', '#', '0', '0', '#', '0', '#', '#']),
            quality=0.42
        )

        mark = PMark()
        mark[0].update(['0', '1'])
        mark[1].update(['0', '1'])
        mark[4].update(['0', '1'])
        mark[6].update(['0', '1'])
        mark[7].update(['0', '1'])

        self.cls.mark = mark

        p0 = Perception(['1', '1', '1', '1', '0', '1', '1', '1'])
        p1 = Perception(['1', '0', '0', '0', '0', '0', '0', '1'])
        time = 684

        # When
        new_cls = self.cls.unexpected_case(p0, p1, time)

        # Then
        self.assertEqual(Condition(['#', '1', '1', '1', '#', '1', '1', '#']),
                         new_cls.condition)
        self.assertEqual(Effect(['#', '0', '0', '0', '#', '0', '0', '#']),
                         new_cls.effect)
        self.assertEqual(mark, self.cls.mark)
        self.assertAlmostEqual(0.39, self.cls.q, places=1)

    def test_should_handle_unexpected_case_5(self):
        # Given
        self.cls = Classifier(
            condition=['0', '0', '#', '#', '#', '#', '1', '#'],
            action=2,
            effect=['#', '#', '#', '#', '#', '#', '#', '#'],
            quality=0.129,
            reward=341.967,
            intermediate_reward=130.369,
            experience=201,
            tga=129,
            talp=9628,
            tav=25.08
        )
        self.cls.mark[2] = '2'
        self.cls.mark[3] = '1'
        self.cls.mark[4] = '1'
        self.cls.mark[5] = '0'
        self.cls.mark[7] = '0'

        p0 = Perception(['0', '0', '2', '1', '1', '0', '1', '0'])
        p1 = Perception(['0', '0', '0', '0', '1', '1', '1', '0'])
        time = 9628

        # When
        new_cls = self.cls.unexpected_case(p0, p1, time)

        # Then
        self.assertIsNotNone(new_cls)
        self.assertEqual(Condition(['0', '0', '2', '1', '#', '0', '1', '#']),
                         new_cls.condition)
        self.assertEqual(Effect(['#', '#', '0', '0', '#', '1', '#', '#']),
                         new_cls.effect)

        self.assertAlmostEqual(0.5, new_cls.q, places=1)
        self.assertAlmostEqual(341.967, new_cls.r, places=1)
        self.assertAlmostEqual(130.369, new_cls.ir, places=1)
        self.assertAlmostEqual(25.08, new_cls.tav, places=1)
        self.assertEqual(1, new_cls.exp)
        self.assertEqual(1, new_cls.num)
        self.assertEqual(time, new_cls.tga)
        self.assertEqual(time, new_cls.talp)

    def test_should_handle_unexpected_case_6(self):
        # Given
        self.cls = Classifier(
            condition=['0', '#', '1', '#', '#', '#', '#', '1'],
            action=2,
            effect=['1', '#', '0', '#', '#', '#', '#', '0'],
            quality=0.38505,
            reward=1.20898,
            intermediate_reward=0,
            experience=11,
            tga=95,
            talp=873,
            tav=71.3967
        )
        self.cls.mark[1].update(['1'])
        self.cls.mark[3].update(['1'])
        self.cls.mark[4].update(['0', '1'])
        self.cls.mark[5].update(['1'])
        self.cls.mark[6].update(['0', '1'])

        p0 = Perception(['0', '1', '1', '1', '1', '1', '0', '1'])
        p1 = Perception(['1', '1', '0', '1', '1', '1', '1', '0'])
        time = 873

        # When
        new_cls = self.cls.unexpected_case(p0, p1, time)

        # Then
        self.assertIsNotNone(new_cls)
        self.assertEqual(Condition(['0', '#', '1', '#', '#', '#', '0', '1']),
                         new_cls.condition)
        self.assertEqual(Effect(['1', '#', '0', '#', '#', '#', '1', '0']),
                         new_cls.effect)

        self.assertAlmostEqual(0.5, new_cls.q, places=1)
        self.assertAlmostEqual(1.20898, new_cls.r, places=1)
        self.assertAlmostEqual(0, new_cls.ir, places=1)
        self.assertAlmostEqual(71.3967, new_cls.tav, places=1)
        self.assertEqual(1, new_cls.exp)
        self.assertEqual(1, new_cls.num)
        self.assertEqual(time, new_cls.tga)
        self.assertEqual(time, new_cls.talp)

    def test_copy_from_and_mutate_does_not_influence_another_condition(self):
        """ Verify that not just reference to Condition copied (changing which
        will change the original - definitily not original C++ code did). """
        operation_time = 123
        original_cl = Classifier(condition=Condition('1###1011'))

        copied_cl = Classifier.copy_from(original_cl, operation_time)

        copied_cl.mutate(TestRandom([s, b, b, b, b]))
        self.assertEqual(Condition('####1011'), copied_cl.condition)
        self.assertEqual(Condition('1###1011'), original_cl.condition)

        original_cl.mutate(TestRandom([b, s, b, b, b]))
        self.assertEqual(Condition('1####011'), original_cl.condition)
        self.assertEqual(Condition('####1011'), copied_cl.condition)

    def test_copy_from_and_change_does_not_influence_another_effect(self):
        """ Verify that not just reference to Condition copied (changing which
        will change the original - definitily not original C++ code did). """
        operation_time = 123
        original_cl = Classifier(effect=Effect('10####1#'))

        copied_cl = Classifier.copy_from(original_cl, operation_time)

        copied_cl.effect[2] = '1'
        self.assertEqual(Effect('101###1#'), copied_cl.effect)
        self.assertEqual(Effect('10####1#'), original_cl.effect)

        original_cl.effect[3] = '0'
        self.assertEqual(Effect('101###1#'), copied_cl.effect)
        self.assertEqual(Effect('10#0##1#'), original_cl.effect)

    def test_should_copy_classifier(self):
        operation_time = 123
        original_cl = Classifier(
            condition=['1', '#', '#', '#', '1', '0', '1', '1'],
            action=1,
            effect=['1', '0', '#', '#', '#', '#', '1', '#'],
            reward=50,
            quality=0.7
        )

        copied_cl = Classifier.copy_from(original_cl, operation_time)

        # Assert that we are dealing with different object
        self.assertFalse(original_cl is copied_cl)

        # Assert that condition is equal but points to another object
        self.assertTrue(original_cl.condition == copied_cl.condition)
        self.assertFalse(original_cl.condition is copied_cl.condition)

        # Assert that action is equal
        self.assertTrue(original_cl.action == copied_cl.action)

        # Assert that effect is equal but points to another object
        self.assertTrue(original_cl.effect == copied_cl.effect)
        self.assertFalse(original_cl.effect is copied_cl.effect)

        # Assert that other properties were set accordingly
        self.assertTrue(copied_cl.mark.is_empty())
        self.assertEqual(50, copied_cl.r)
        self.assertEqual(0.7, copied_cl.q)
        self.assertEqual(operation_time, copied_cl.tga)
        self.assertEqual(operation_time, copied_cl.talp)

    def test_should_detect_similar_classifiers_1(self):
        # Create baseline classifier
        base = Classifier(
            condition=['1', '#', '#', '#', '1', '0', '1', '1'],
            action=1,
            effect=['1', '0', '#', '#', '#', '#', '1', '#']
        )

        # Test two similar classifiers
        c1 = Classifier(
            condition=['1', '#', '#', '#', '1', '0', '1', '1'],
            action=1,
            effect=['1', '0', '#', '#', '#', '#', '1', '#']
        )
        self.assertTrue(base.is_similar(c1))

    def test_similar_returns_true_if_differs_by_numbers(self):
        original = Classifier(
            condition=Condition('#01##10#'),
            action=2,
            effect=Effect('1##01##0'),
            numerosity=1.1,
            experience=0.9,
            intermediate_reward=1.2,
            quality=0.5,
            reward=0.6,
            talp=None,
            tav=1,
            tga=2
        )
        c_num = Classifier(
            condition=Condition('#01##10#'),
            action=2,
            effect=Effect('1##01##0'),
            numerosity=1.2,
            experience=0.9,
            intermediate_reward=1.2,
            quality=0.5,
            reward=0.6,
            talp=None,
            tav=1,
            tga=2
        )
        c_exp = Classifier(
            condition=Condition('#01##10#'),
            action=2,
            effect=Effect('1##01##0'),
            numerosity=1.1,
            experience=0.95,
            intermediate_reward=1.2,
            quality=0.5,
            reward=0.6,
            talp=None,
            tav=1,
            tga=2
        )
        c_inter = Classifier(
            condition=Condition('#01##10#'),
            action=2,
            effect=Effect('1##01##0'),
            numerosity=1.1,
            experience=0.9,
            intermediate_reward=1.3,
            quality=0.5,
            reward=0.6,
            talp=None,
            tav=1,
            tga=2
        )
        c_qual = Classifier(
            condition=Condition('#01##10#'),
            action=2,
            effect=Effect('1##01##0'),
            numerosity=1.1,
            experience=0.9,
            intermediate_reward=1.2,
            quality=1,
            reward=0.6,
            talp=None,
            tav=1,
            tga=2
        )
        c_rew = Classifier(
            condition=Condition('#01##10#'),
            action=2,
            effect=Effect('1##01##0'),
            numerosity=1.1,
            experience=0.9,
            intermediate_reward=1.2,
            quality=0.5,
            reward=0.5,
            talp=None,
            tav=1,
            tga=2
        )
        c_talp = Classifier(
            condition=Condition('#01##10#'),
            action=2,
            effect=Effect('1##01##0'),
            numerosity=1.1,
            experience=0.9,
            intermediate_reward=1.2,
            quality=0.5,
            reward=0.6,
            talp=1,
            tav=1,
            tga=2
        )
        c_tav = Classifier(
            condition=Condition('#01##10#'),
            action=2,
            effect=Effect('1##01##0'),
            numerosity=1.1,
            experience=0.9,
            intermediate_reward=1.2,
            quality=0.5,
            reward=0.6,
            talp=None,
            tav=2,
            tga=2
        )
        c_tga = Classifier(
            condition=Condition('#01##10#'),
            action=2,
            effect=Effect('1##01##0'),
            numerosity=1.1,
            experience=0.9,
            intermediate_reward=1.2,
            quality=0.5,
            reward=0.6,
            talp=None,
            tav=1,
            tga=0
        )
        self.assertTrue(c_num.is_similar(original))
        self.assertTrue(c_exp.is_similar(original))
        self.assertTrue(c_inter.is_similar(original))
        self.assertTrue(c_qual.is_similar(original))
        self.assertTrue(c_rew.is_similar(original))
        self.assertTrue(c_talp.is_similar(original))
        self.assertTrue(c_tav.is_similar(original))
        self.assertTrue(c_tga.is_similar(original))
        self.assertTrue(original.is_similar(c_num))
        self.assertTrue(original.is_similar(c_exp))
        self.assertTrue(original.is_similar(c_inter))
        self.assertTrue(original.is_similar(c_qual))
        self.assertTrue(original.is_similar(c_rew))
        self.assertTrue(original.is_similar(c_talp))
        self.assertTrue(original.is_similar(c_tav))
        self.assertTrue(original.is_similar(c_tga))

    def test_should_detect_similar_classifiers_2(self):
        # Create baseline classifier
        base = Classifier(
            condition=['1', '#', '#', '#', '1', '0', '1', '1'],
            action=1,
            effect=['1', '0', '#', '#', '#', '#', '1', '#']
        )

        # Changed condition part
        self.assertFalse(base.is_similar(
            Classifier(
                condition=['1', '#', '1', '#', '1', '0', '1', '1'],
                action=1,
                effect=['1', '0', '#', '#', '#', '#', '1', '#']
            )))

        # Changed action part
        self.assertFalse(base.is_similar(
            Classifier(
                condition=['1', '#', '#', '#', '1', '0', '1', '1'],
                action=2,
                effect=['1', '0', '#', '#', '#', '#', '1', '#']
            )))

        # Changed effect part
        self.assertFalse(base.is_similar(
            Classifier(
                condition=['1', '#', '#', '#', '1', '0', '1', '1'],
                action=1,
                effect=['1', '0', '#', '#', '#', '#', '1', '1']
            )))

    def test_should_detect_more_general_classifier_1(self):
        # No specified elements - should not be more general

        # Given
        c = Classifier()

        # When
        res = self.cls.is_more_general(c)

        # Then
        self.assertFalse(res)

    def test_should_detect_more_general_classifier_2(self):
        # Should be more general
        c = Classifier(condition=['1', '#', '#', '#', '1', '0', '1', '1'])
        self.assertTrue(self.cls.is_more_general(c))

    def test_should_detect_more_general_classifier_3(self):
        # Shouldn't be more general
        c = Classifier(condition=['1', '#', '#', '#', '1', '#', '#', '#'])
        self.cls.condition = Condition(
            ['1', '#', '1', '#', '1', '0', '1', '1'])
        self.assertFalse(self.cls.is_more_general(c))

    def test_should_distinguish_classifier_as_subsumer_1(self):
        # General classifier should not be considered as subsumer
        self.assertFalse(self.cls._is_subsumer())

    def test_should_distinguish_classifier_as_subsumer_2(self):
        # Let's assign enough experience and quality
        self.cls.exp = 30
        self.cls.q = 0.92
        self.assertTrue(self.cls._is_subsumer())

    def test_should_distinguish_classifier_as_subsumer_3(self):
        # Let's reduce experience below threshold
        self.cls.exp = 15
        self.cls.q = 0.92
        self.assertFalse(self.cls._is_subsumer())

    def test_should_distinguish_classifier_as_subsumer_4(self):
        # Now check if the fact that classifier is marked will block
        # it from being considered as a subsumer
        self.cls.exp = 30
        self.cls.q = 0.92
        self.cls.mark[3] = '1'
        self.assertFalse(self.cls._is_subsumer())

    def test_should_subsume_another_classifier_1(self):
        self.cls.condition[3] = '0'
        self.cls.action = 3
        self.cls.effect[2] = '1'
        self.cls.q = 0.93
        self.cls.r = 1.35
        self.cls.exp = 23

        other = Classifier()
        other.condition[0] = '1'
        other.condition[3] = '0'
        other.action = 3
        other.effect[2] = '1'
        other.q = 0.5
        other.r = 0.35
        other.exp = 1

        self.assertTrue(self.cls.does_subsume(other))

    def test_should_subsume_another_classifier_2(self):
        self.cls.condition[0] = '1'
        self.cls.condition[1] = '0'
        self.cls.condition[4] = '0'
        self.cls.condition[6] = '1'
        self.cls.action = 6
        self.cls.effect[0] = '0'
        self.cls.effect[1] = '1'
        self.cls.effect[6] = '0'
        self.cls.q = 0.84
        self.cls.r = 0.33
        self.cls.exp = 3

        other = Classifier()
        other.condition[0] = '1'
        other.condition[1] = '0'
        other.condition[6] = '2'
        other.action = 3
        other.effect[0] = '0'
        other.effect[1] = '1'
        other.effect[6] = '0'
        other.q = 0.5
        other.r = 0.41
        other.exp = 1

        self.assertFalse(self.cls.does_subsume(other))

    def test_should_subsume_another_classifier_3(self):
        # Given
        self.cls.condition[6] = '0'
        self.cls.action = 6
        self.cls.q = 0.99
        self.cls.r = 11.4
        self.cls.exp = 32

        other = Classifier()
        other.condition[3] = '1'
        other.condition[6] = '0'
        other.action = 6
        other.q = 0.5
        other.r = 9.89
        other.exp = 1

        # When, Then
        self.assertTrue(self.cls.does_subsume(other))

    def test_should_set_mark_from_condition_1(self):
        # Given
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        self.cls.condition = Condition(
            ['#', '#', '0', '#', '1', '#', '1', '#'])
        self.cls.mark[0] = '0'
        self.cls.mark[1] = '0'
        self.cls.mark[3] = '0'
        self.cls.mark[5] = '1'
        self.cls.mark[7] = '1'

        # When
        self.cls.set_mark(p0)

        # Then
        self.assertEqual(5, len(self.cls.mark))
        self.assertEqual(1, len(self.cls.mark[0]))  # 0
        self.assertEqual(1, len(self.cls.mark[1]))  # 0
        self.assertEqual(0, len(self.cls.mark[2]))
        self.assertEqual(1, len(self.cls.mark[3]))  # 0
        self.assertEqual(0, len(self.cls.mark[4]))
        self.assertEqual(1, len(self.cls.mark[5]))  # 1
        self.assertEqual(0, len(self.cls.mark[6]))
        self.assertEqual(1, len(self.cls.mark[7]))  # 1

    def test_should_set_mark_from_condition_2(self):
        # Given
        p0 = Perception(['1', '2', '1', '0', '1', '1', '0', '1'])
        self.cls.condition = Condition(
            ['#', '#', '#', '0', '#', '1', '0', '1'])

        # When
        self.cls.set_mark(p0)

        # Then
        self.assertEqual(4, len(self.cls.mark))

        self.assertEqual(1, len(self.cls.mark[0]))
        self.assertIn('1', self.cls.mark[0])

        self.assertEqual(1, len(self.cls.mark[1]))
        self.assertIn('2', self.cls.mark[1])

        self.assertEqual(1, len(self.cls.mark[2]))
        self.assertIn('1', self.cls.mark[2])

        self.assertEqual(1, len(self.cls.mark[4]))
        self.assertIn('1', self.cls.mark[4])

    def test_should_set_mark_from_condition_3(self):
        # Given
        p0 = Perception(['1', '1', '1', '1', '1', '0', '1', '0'])
        self.cls.condition = Condition(
            ['1', '1', '#', '1', '1', '#', '#', '0'])

        # When
        self.cls.set_mark(p0)

        # Then
        self.assertEqual(3, len(self.cls.mark))

        self.assertEqual(1, len(self.cls.mark[2]))
        self.assertIn('1', self.cls.mark[2])

        self.assertEqual(1, len(self.cls.mark[5]))
        self.assertIn('0', self.cls.mark[5])

        self.assertEqual(1, len(self.cls.mark[6]))
        self.assertIn('1', self.cls.mark[6])

    def test_should_set_mark_from_condition_4(self):
        # Given
        p0 = Perception(['0', '1', '1', '0', '0', '0', '0', '0'])
        self.cls.condition = Condition(
            ['#', '#', '#', '0', '#', '#', '#', '0'])

        # When
        self.cls.set_mark(p0)

        # Then
        self.assertEqual(6, len(self.cls.mark))

        self.assertEqual(1, len(self.cls.mark[0]))
        self.assertIn('0', self.cls.mark[0])

        self.assertEqual(1, len(self.cls.mark[1]))
        self.assertIn('1', self.cls.mark[1])

        self.assertEqual(1, len(self.cls.mark[2]))
        self.assertIn('1', self.cls.mark[2])

        self.assertEqual(1, len(self.cls.mark[4]))
        self.assertIn('0', self.cls.mark[4])

        self.assertEqual(1, len(self.cls.mark[5]))
        self.assertIn('0', self.cls.mark[5])

        self.assertEqual(1, len(self.cls.mark[6]))
        self.assertIn('0', self.cls.mark[6])

    def test_should_predict_successfully_1(self):
        # Given
        self.cls = Classifier(
            condition=['1', '#', '0', '1', '1', '1', '#', '1'],
            action=5,
            effect=['0', '#', '1', '0', '0', '0', '#', '0'],
            quality=0.94
        )
        action = 5
        p0 = Perception(['1', '1', '0', '1', '1', '1', '0', '1'])
        p1 = Perception(['0', '1', '1', '0', '0', '0', '0', '0'])

        # Then
        self.assertTrue(self.cls.predicts_successfully(p0, action, p1))
