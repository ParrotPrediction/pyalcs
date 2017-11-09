import unittest

from alcs import Perception
from alcs.acs2 import Classifier, Condition, Effect, PMark


class ClassifierTest(unittest.TestCase):

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
