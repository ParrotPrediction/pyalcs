import pytest

from lcs import Perception
from lcs.agents.acs2 import Configuration, Classifier, \
    Condition, Effect


class TestClassifier:

    @pytest.fixture
    def cfg(self):
        return Configuration(8, 8)

    @pytest.mark.parametrize("_c1, _a1, _e1, _c2, _a2, _e2, _result", [
        # Similar classifiers - everything the same
        ('00001111', 1, '0000111#', '00001111', 1, '0000111#', True),
        # Different condition
        ('00001111', 1, '0000111#', '000#1111', 1, '0000111#', False),
        # Different action
        ('00001111', 1, '0000111#', '00001111', 2, '0000111#', False),
        # Different effect
        ('00001111', 1, '0000111#', '00001111', 1, '00001111', False),
    ])
    def test_equality(self, _c1, _a1, _e1, _c2, _a2, _e2, _result, cfg):
        # given
        cl1 = Classifier(condition=_c1, action=_a1, effect=_e1, cfg=cfg)
        cl2 = Classifier(condition=_c2, action=_a2, effect=_e2, cfg=cfg)

        # then
        assert (cl1 == cl2) is _result

    def test_should_calculate_fitness(self, cfg):
        # given
        cls = Classifier(reward=0.25, cfg=cfg)

        # then
        assert cls.fitness == 0.125

    def test_should_anticipate_change(self, cfg):
        # given
        cls = Classifier(cfg=cfg)
        assert cls.does_anticipate_change() is False

        # when
        cls.effect[1] = '1'

        # then
        assert cls.does_anticipate_change() is True

    def test_should_anticipate_correctly(self, cfg):
        # given
        cls = Classifier(
            effect='#1####0#',
            cfg=cfg)
        p0 = Perception('00001111')
        p1 = Perception('01001101')

        # then
        assert cls.does_anticipate_correctly(p0, p1) is True

    @pytest.mark.parametrize("_condition, _specificity", [
        ('########', 0.0),
        ('#1#01#0#', 0.5),
        ('11101001', 1.0),
    ])
    def test_should_calculate_specificity(
            self, _condition, _specificity, cfg):

        cls = Classifier(Condition(_condition), cfg=cfg)
        assert cls.specificity == _specificity

    def test_should_be_considered_as_reliable_1(self, cfg):
        # given
        cls = Classifier(quality=0.89, cfg=cfg)

        # then
        assert cls.is_reliable() is False

    def test_should_be_considered_as_reliable_2(self, cfg):
        # given
        cls = Classifier(quality=0.91, cfg=cfg)

        # then
        assert cls.is_reliable() is True

    def test_should_be_considered_as_inadequate_1(self, cfg):
        # given
        cls = Classifier(quality=0.50, cfg=cfg)

        # then
        assert cls.is_reliable() is False

    def test_should_be_considered_as_inadequate_2(self, cfg):
        # given
        cls = Classifier(quality=0.09, cfg=cfg)

        # then
        assert cls.is_inadequate() is True

    def test_should_increase_experience(self, cfg):
        # given
        cls = Classifier(experience=5, cfg=cfg)

        # when
        cls.increase_experience()

        # then
        assert 6 == cls.exp

    def test_should_increase_quality(self, cfg):
        # given
        cls = Classifier(quality=0.5, cfg=cfg)

        # when
        cls.increase_quality()

        # then
        assert 0.525 == cls.q

    def test_should_decrease_quality(self, cfg):
        # given
        cls = Classifier(quality=0.47, cfg=cfg)

        # when
        cls.decrease_quality()

        # then
        assert abs(0.45 - cls.q) < 0.01

    def test_should_detect_correct_anticipation_1(self, cfg):
        # Classifier is not predicting any change, all pass-through effect
        # should predict correctly

        # given
        cls = Classifier(effect=Effect('########'), cfg=cfg)
        p0 = Perception('00001111')
        p1 = Perception('00001111')

        # then
        assert cls.does_anticipate_correctly(p0, p1) is True

    def test_should_detect_correct_anticipation_2(self, cfg):
        # Introduce two changes into situation and effect (should
        # also predict correctly)

        # given
        cls = Classifier(
            effect=Effect(['#', '1', '#', '#', '#', '#', '0', '#']),
            cfg=cfg)
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '1', '0', '0', '1', '1', '0', '1'])

        # then
        assert cls.does_anticipate_correctly(p0, p1) is True

    def test_should_detect_correct_anticipation_3(self, cfg):
        # Case when effect predicts situation incorrectly

        # given
        cls = Classifier(
            effect=Effect(['#', '0', '#', '#', '#', '#', '#', '#']),
            cfg=cfg)
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '1', '0', '0', '1', '1', '1', '1'])

        # then
        assert cls.does_anticipate_correctly(p0, p1) is False

    def test_should_detect_correct_anticipation_4(self, cfg):
        # Case when effect predicts situation incorrectly

        # given
        cls = Classifier(
            effect=Effect(['#', '#', '0', '#', '#', '1', '#', '#']),
            cfg=cfg)
        p0 = Perception(['1', '0', '1', '0', '1', '0', '0', '1'])
        p1 = Perception(['1', '0', '1', '0', '1', '0', '0', '1'])

        # then
        assert cls.does_anticipate_correctly(p0, p1) is False

    def test_should_detect_correct_anticipation_5(self, cfg):
        # Case when effect predicts situation incorrectly

        # given
        cls = Classifier(
            effect=Effect(['#', '#', '#', '#', '1', '#', '0', '#']),
            cfg=cfg)
        p0 = Perception(['0', '1', '1', '0', '0', '0', '1', '1'])
        p1 = Perception(['1', '1', '1', '0', '1', '1', '0', '1'])

        # then
        assert cls.does_anticipate_correctly(p0, p1) is False

    def test_should_detect_correct_anticipation_6(self, cfg):
        # Case when effect predicts situation incorrectly

        # given
        cls = Classifier(
            effect=Effect(['#', '#', '1', '#', '0', '#', '0', '#']),
            cfg=cfg)
        p0 = Perception(['0', '0', '0', '1', '1', '0', '1', '0'])
        p1 = Perception(['0', '0', '1', '1', '0', '0', '0', '0'])

        # then
        assert cls.does_anticipate_correctly(p0, p1) is True

    def test_should_handle_pass_through_symbol(self, cfg):
        # A case when there was no change in perception but effect has no
        # pass-through symbol

        # given
        cls = Classifier(
            effect=Effect(['#', '0', '#', '#', '#', '#', '#', '#']),
            cfg=cfg)
        p0 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])
        p1 = Perception(['0', '0', '0', '0', '1', '1', '1', '1'])

        # then
        assert cls.does_anticipate_correctly(p0, p1) is False

    @pytest.mark.parametrize("_p0, _p1,"
                             "_init_cond, _init_effect,"
                             "_res_cond, _res_effect", [
                                 ('00001111', '00001111',
                                  '########', '########',
                                  '########', '########'),
                                 ('00001111', '00011111',
                                  '########', '########',
                                  '###0####', '###1####'),
                                 ('01110111', '10101010',
                                  '01#####1', '10#####0',
                                  '01#101#1', '10#010#0')])
    def test_should_specialize(self,
                               _p0, _p1,
                               _init_cond, _init_effect,
                               _res_cond, _res_effect, cfg):
        # given
        cls = Classifier(
            condition=Condition(_init_cond),
            effect=Effect(_init_effect),
            cfg=cfg)
        p0 = Perception(_p0)
        p1 = Perception(_p1)

        # when
        cls.specialize(p0, p1, leave_specialized=False)

        # then
        assert cls.condition == Condition(_res_cond)
        assert cls.effect == Effect(_res_effect)

    @pytest.mark.parametrize("_p0, _p1,"
                             "_init_cond, _init_effect,"
                             "_res_cond, _res_effect", [
                                 ('00001111', '00001111',
                                  '########', '########',
                                  '########', '########'),
                                 ('00001111', '00001111',
                                  '########', '1#######',
                                  '########', '1#######'),
                             ])
    def test_should_specialize_wrt_wildcards(self,
                                             _p0, _p1,
                                             _init_cond, _init_effect,
                                             _res_cond, _res_effect, cfg):
        # given
        cls = Classifier(
            condition=Condition(_init_cond),
            effect=Effect(_init_effect),
            cfg=cfg)
        p0 = Perception(_p0)
        p1 = Perception(_p1)

        # when
        cls.specialize(p0, p1, leave_specialized=True)

        # then
        assert cls.condition == Condition(_res_cond)
        assert cls.effect == Effect(_res_effect)

    @pytest.mark.parametrize("_condition, _effect, _sua", [
        ('######0#', '########', 1),
        ('#####0#0', '########', 2),
        ('1###1011', '0####1##', 3),
        ('1#0#1011', '0####1##', 4),
        ('10000001', '####1#1#', 6),
    ])
    def test_should_count_specified_unchanging_attributes(
            self, _condition, _effect, _sua, cfg):

        # given
        cls = Classifier(condition=_condition, effect=_effect, cfg=cfg)

        # when & then
        assert len(cls.specified_unchanging_attributes) == _sua

    def test_copy_from_and_change_does_not_influence_another_effect(self, cfg):
        """ Verify that not just reference to Condition copied (changing which
        will change the original - definitily not original C++ code did). """
        # given
        operation_time = 123
        original_cl = Classifier(
            effect='10####1#',
            cfg=cfg)

        # when
        copied_cl = Classifier.copy_from(original_cl, operation_time)

        # when & then
        copied_cl.effect[2] = '1'
        assert Effect('101###1#') == copied_cl.effect
        assert Effect('10####1#') == original_cl.effect

        # when & then
        original_cl.effect[3] = '0'
        assert Effect('101###1#') == copied_cl.effect
        assert Effect('10#0##1#') == original_cl.effect

    def test_should_copy_classifier(self, cfg):
        # given
        operation_time = 123
        original_cl = Classifier(
            condition='1###1011',
            action=1,
            effect='10####1#',
            reward=50,
            quality=0.7,
            cfg=cfg
        )

        # when
        copied_cl = Classifier.copy_from(original_cl, operation_time)

        # Assert that we are dealing with different object
        assert original_cl is not copied_cl

        # Assert that condition is equal but points to another object
        assert original_cl.condition == copied_cl.condition
        assert original_cl.condition is not copied_cl.condition

        # Assert that action is equal
        assert original_cl.action == copied_cl.action

        # Assert that effect is equal but points to another object
        assert original_cl.effect == copied_cl.effect
        assert original_cl.effect is not copied_cl.effect

        # Assert that other properties were set accordingly
        assert copied_cl.is_marked() is False
        assert 50 == copied_cl.r
        assert 0.7 == copied_cl.q
        assert operation_time == copied_cl.tga
        assert operation_time == copied_cl.talp

    def test_should_detect_similar_classifiers_1(self, cfg):
        # given
        base = Classifier(
            condition='1###1011',
            action=1,
            effect='10####1#',
            cfg=cfg
        )

        c1 = Classifier(
            condition='1###1011',
            action=1,
            effect='10####1#',
            cfg=cfg
        )

        # when && then
        assert base == c1

    def test_similar_returns_true_if_differs_by_numbers(self, cfg):
        # given
        original = Classifier(
            condition='#01##10#',
            action=2,
            effect='1##01##0',
            numerosity=1.1,
            experience=0.9,
            immediate_reward=1.2,
            quality=0.5,
            reward=0.6,
            talp=None,
            tav=1,
            tga=2,
            cfg=cfg
        )

        c_num = Classifier(
            condition='#01##10#',
            action=2,
            effect='1##01##0',
            numerosity=1.2,
            experience=0.9,
            immediate_reward=1.2,
            quality=0.5,
            reward=0.6,
            talp=None,
            tav=1,
            tga=2,
            cfg=cfg
        )

        c_exp = Classifier(
            condition='#01##10#',
            action=2,
            effect='1##01##0',
            numerosity=1.1,
            experience=0.95,
            immediate_reward=1.2,
            quality=0.5,
            reward=0.6,
            talp=None,
            tav=1,
            tga=2,
            cfg=cfg
        )

        c_inter = Classifier(
            condition='#01##10#',
            action=2,
            effect='1##01##0',
            numerosity=1.1,
            experience=0.9,
            immediate_reward=1.3,
            quality=0.5,
            reward=0.6,
            talp=None,
            tav=1,
            tga=2,
            cfg=cfg
        )

        c_qual = Classifier(
            condition='#01##10#',
            action=2,
            effect='1##01##0',
            numerosity=1.1,
            experience=0.9,
            immediate_reward=1.2,
            quality=1,
            reward=0.6,
            talp=None,
            tav=1,
            tga=2,
            cfg=cfg
        )

        c_rew = Classifier(
            condition='#01##10#',
            action=2,
            effect='1##01##0',
            numerosity=1.1,
            experience=0.9,
            immediate_reward=1.2,
            quality=0.5,
            reward=0.5,
            talp=None,
            tav=1,
            tga=2,
            cfg=cfg
        )

        c_talp = Classifier(
            condition='#01##10#',
            action=2,
            effect='1##01##0',
            numerosity=1.1,
            experience=0.9,
            immediate_reward=1.2,
            quality=0.5,
            reward=0.6,
            talp=1,
            tav=1,
            tga=2,
            cfg=cfg
        )

        c_tav = Classifier(
            condition='#01##10#',
            action=2,
            effect='1##01##0',
            numerosity=1.1,
            experience=0.9,
            immediate_reward=1.2,
            quality=0.5,
            reward=0.6,
            talp=None,
            tav=2,
            tga=2,
            cfg=cfg
        )

        c_tga = Classifier(
            condition='#01##10#',
            action=2,
            effect='1##01##0',
            numerosity=1.1,
            experience=0.9,
            immediate_reward=1.2,
            quality=0.5,
            reward=0.6,
            talp=None,
            tav=1,
            tga=0,
            cfg=cfg
        )

        # then
        cls = [c_num, c_exp, c_inter, c_qual, c_rew, c_talp, c_tav, c_tga]
        for cl in cls:
            assert cl == original

    def test_should_detect_similar_classifiers_2(self, cfg):
        # given
        base = Classifier(
            condition='1###1011',
            action=1,
            effect='10####1#',
            cfg=cfg
        )

        # when & then
        # Changed condition part
        assert base != (
            Classifier(
                condition='1#1#1011',
                action=1,
                effect='10####1#',
                cfg=cfg
            ))

        # when & then
        # changed action part
        assert base != (
            Classifier(
                condition='1###1011',
                action=2,
                effect='10####1#',
                cfg=cfg
            ))

        # when & then
        # changed effect part
        assert base != (
            Classifier(
                condition='1###1011',
                action=1,
                effect='10####11',
                cfg=cfg
            ))

    @pytest.mark.parametrize("_c1_condition, _c2_condition, _result", [
        (None, None, False),
        (None, '1###1011', True),
        ('1#1#1011', '1###1###', False)
    ])
    def test_should_detect_more_general_classifier(
            self, _c1_condition, _c2_condition, _result, cfg):

        # given
        c1 = Classifier(condition=_c1_condition, cfg=cfg)
        c2 = Classifier(condition=_c2_condition, cfg=cfg)

        # when & then
        assert c1.is_more_general(c2) is _result

    def test_should_set_mark_from_condition_1(self, cfg):
        # given
        p0 = Perception('00001111')
        cls = Classifier(condition='##0#1#1#', cfg=cfg)
        cls.mark[0].add('0')
        cls.mark[1].add('0')
        cls.mark[3].add('0')
        cls.mark[5].add('1')
        cls.mark[7].add('1')

        # when
        cls.set_mark(p0)

        # then
        assert 8 == len(cls.mark)
        assert 1 == len(cls.mark[0])  # 0
        assert 1 == len(cls.mark[1])  # 0
        assert 0 == len(cls.mark[2])
        assert 1 == len(cls.mark[3])  # 0
        assert 0 == len(cls.mark[4])
        assert 1 == len(cls.mark[5])  # 1
        assert 0 == len(cls.mark[6])
        assert 1 == len(cls.mark[7])  # 1

    def test_should_set_mark_from_condition_2(self, cfg):
        # given
        p0 = Perception('12101101')
        cls = Classifier(condition='###0#101', cfg=cfg)

        # when
        cls.set_mark(p0)

        # then
        assert 1 == len(cls.mark[0])
        assert '1' in cls.mark[0]

        assert 1 == len(cls.mark[1])
        assert '2' in cls.mark[1]

        assert 1 == len(cls.mark[2])
        assert '1' in cls.mark[2]

        assert 1 == len(cls.mark[4])
        assert '1' in cls.mark[4]

    def test_should_set_mark_from_condition_3(self, cfg):
        # given
        p0 = Perception('11111010')
        cls = Classifier(condition='11#11##0', cfg=cfg)

        # when
        cls.set_mark(p0)

        # Then
        assert 1 == len(cls.mark[2])
        assert '1' in cls.mark[2]

        assert 1 == len(cls.mark[5])
        assert '0' in cls.mark[5]

        assert 1 == len(cls.mark[6])
        assert '1' in cls.mark[6]

    def test_should_set_mark_from_condition_4(self, cfg):
        # given
        p0 = Perception('01100000')
        cls = Classifier(condition='###0###0', cfg=cfg)

        # when
        cls.set_mark(p0)

        # then
        assert 1 == len(cls.mark[0])
        assert '0' in cls.mark[0]

        assert 1 == len(cls.mark[1])
        assert '1' in cls.mark[1]

        assert 1 == len(cls.mark[2])
        assert '1' in cls.mark[2]

        assert 1 == len(cls.mark[4])
        assert '0' in cls.mark[4]

        assert 1 == len(cls.mark[5])
        assert '0' in cls.mark[5]

        assert 1 == len(cls.mark[6])
        assert '0' in cls.mark[6]

    def test_should_predict_successfully_1(self, cfg):
        # given
        action = 5
        cls = Classifier(
            condition='1#0111#1',
            action=action,
            effect='0#1000#0',
            quality=0.94,
            cfg=cfg
        )
        p0 = Perception('11011101')
        p1 = Perception('01100000')

        # then
        assert cls.predicts_successfully(p0, action, p1) is True

    @pytest.mark.parametrize("_condition, _effect, _sua_before, _sua_after", [
        ('#####0#0', '#####1#1', 0, 0),
        ('#####0#0', '#######1', 1, 0),
        ('#####0#0', '########', 2, 1),
    ])
    def test_should_not_generalize_unchanging_attribute(
            self, _condition, _effect, _sua_before, _sua_after, cfg):

        # given
        cls = Classifier(condition=_condition, effect=_effect, cfg=cfg)
        assert len(cls.specified_unchanging_attributes) == _sua_before

        # when
        cls.generalize_unchanging_condition_attribute()

        # then
        assert len(cls.specified_unchanging_attributes) == _sua_after

    def test_should_generalize_second_unchanging_attribute(self, cfg):
        # given
        cls = Classifier(
            condition='#####0#0',
            effect='########',
            cfg=cfg)

        assert len(cls.specified_unchanging_attributes) == 2

        # when
        generalized = cls.generalize_unchanging_condition_attribute(
            lambda x: 7)

        # then
        assert generalized is True
        assert len(cls.specified_unchanging_attributes) == 1
        assert Condition('#####0##') == cls.condition

    @pytest.mark.parametrize("_e, _result", [
        ('#####1#1', True),
        ('0######1', False),
        ('#####0#1', False)
    ])
    def test_does_match_backwards(self, cfg, _e, _result):
        # given
        percept = Perception('11111111')
        c1 = Classifier(condition='#####0#0', effect=_e, cfg=cfg)

        # when
        result1 = c1.does_match_backwards(percept)

        # then
        assert result1 is _result

    @pytest.mark.parametrize("_c, _e, _result", [
        ('#0###1##', '#1###0##', ['0', '0', '1', '1', '1', '1', '1', '0']),
        ('#0###1##', '#0###0##', None),
        ('#0###1##', '0####0##', None)
    ])
    def test_get_backwards_anticipation(self, cfg, _c, _e, _result):
        # given
        p0 = Perception(['0', '1', '1', '1', '1', '0', '1', '0'])
        c0 = Classifier(condition=_c, effect=_e, cfg=cfg)

        # when
        result0 = c0.get_backwards_anticipation(p0)

        # then
        assert result0 == _result

    @pytest.mark.parametrize("_p0, _result", [
        (['1', '1', '0', '1', '1', '1', '1', '1'],
         ['1', '1', '0', '1', '1', '0', '1', '1']),
        (['1', '1', '1', '1', '1', '1', '1', '1'],
         ['1', '1', '0', '1', '1', '0', '1', '1'])
    ])
    def test_get_best_anticipation(self, cfg, _p0, _result):
        # given
        p0 = Perception(_p0)
        classifier = Classifier(effect='##0##0##', cfg=cfg)

        # when
        result0 = classifier.get_best_anticipation(p0)

        # then
        assert result0 == _result
