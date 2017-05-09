import unittest

from alcs.agent import Perception
from alcs.agent.acs2 import Classifier, Condition, Effect, PMark


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        self.cls = Classifier()

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

    def test_should_update_reward(self):
        self.cls.update_reward(1000)
        self.assertEqual(50.475, self.cls.r)

    def test_should_update_intermediate_reward(self):
        self.cls.update_intermediate_reward(1000)
        self.assertEqual(50.0, self.cls.ir)

    def test_should_increase_quality(self):
        self.cls.q = 0.5
        self.cls.increase_quality()
        self.assertEqual(0.525, self.cls.q)

    def test_should_decrease_quality(self):
        self.cls.q = 0.47
        self.cls.decrease_quality()
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
        p0 = Perception(['0', '1', '1', '1', '1', '1', '0', '1'])
        p1 = Perception(['1', '1', '0', '1', '1', '1', '1', '0'])
        self.cls.effect[0] = '1'
        self.cls.effect[2] = '0'
        self.cls.effect[7] = '0'

        # When
        self.cls.specialize(p0, p1)

        # Then
        self.assertEqual(1, self.cls.condition.specificity)
        self.assertEqual('0', self.cls.condition[6])


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
        self.cls.condition = Condition(['#', '#', '#', '#', '#', '#', '#', '0'])
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

    def test_should_detect_similar_classifiers(self):
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

    def test_should_spot_non_similar_classifiers(self):
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

    def test_should_detect_more_general_classifier(self):
        # No specified elements - should not be more general
        self.assertFalse(self.cls.is_more_general(Classifier()))

        # Should be more general
        c = Classifier(condition=['1', '#', '#', '#', '1', '0', '1', '1'])
        self.assertTrue(self.cls.is_more_general(c))

        # Shouldn't be more general
        c = Classifier(condition=['1', '#', '#', '#', '1', '#', '#', '#'])
        self.cls.condition = Condition(['1', '#', '1', '#', '1', '0', '1', '1'])
        self.assertFalse(self.cls.is_more_general(c))

    def test_should_distinguish_classifier_as_subsumer(self):
        # General classifier should not be considered as subsumer
        self.assertFalse(self.cls._is_subsumer())

        # Let's assign enough experience and quality
        self.cls.exp = 30
        self.cls.q = 0.92
        self.assertTrue(self.cls._is_subsumer())

        # Let's reduce experience below threshold
        self.cls.exp = 15
        self.assertFalse(self.cls._is_subsumer())

        # Now check if the fact that classifier is marked will block
        # it from being considered as a subsumer
        self.cls.exp = 30
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

    def test_should_set_mark_from_condition(self):
        # Given
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        self.cls.condition = Condition(['#', '#', '0', '#', '1', '#', '1', '#'])
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
        self.cls.condition = Condition(['#', '#', '#', '0', '#', '1', '0', '1'])

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
