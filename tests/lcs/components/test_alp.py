import pytest

from lcs import Perception
from lcs.agents.acs2 import Configuration, Classifier, \
    Condition, Effect
from lcs.components.alp import expected_case, unexpected_case, cover


class TestALP:

    @pytest.fixture
    def cfg(self):
        return Configuration(8, 8)

    def test_should_handle_expected_case_1(self, cfg):
        # given
        cls = Classifier(
            condition='#######0',
            quality=0.525,
            cfg=cfg)
        p0 = Perception('11111010')
        time = 47

        # when
        new_cls = expected_case(cls, p0, time)

        # then
        assert new_cls is None
        assert abs(0.54 - cls.q) < 0.01

    def test_should_handle_expected_case_2(self, cfg):
        # given
        cls = Classifier(
            condition='#0######',
            quality=0.521,
            cfg=cfg)
        p0 = Perception('10101001')
        time = 59

        # when
        new_cls = expected_case(cls, p0, time)

        # then
        assert new_cls is None
        assert abs(0.54 - cls.q) < 0.01

    def test_should_handle_expected_case_3(self, cfg):
        # given
        p0 = Perception('00110000')
        time = 26
        cls = Classifier(
            action=5,
            quality=0.46,
            cfg=cfg
        )
        cls.mark[0] = '0'
        cls.mark[1] = '1'
        cls.mark[2] = '0'
        cls.mark[3] = '1'
        cls.mark[4] = '0'
        cls.mark[5] = '1'
        cls.mark[6] = '1'
        cls.mark[7] = '1'

        # when
        new_cls = expected_case(cls, p0, time)

        # then
        assert new_cls is not None
        # One `random` attribute gets specified
        assert 1 == new_cls.condition.specificity
        assert Effect('########') == new_cls.effect
        assert 5 == new_cls.action
        assert new_cls.mark.is_empty() is True
        assert 0.5 == new_cls.q

    def test_should_handle_expected_case_4(self, cfg):
        # given
        p0 = Perception('11101101')
        time = 703
        cls = Classifier(
            condition='1##01#0#',
            action=7,
            effect='0##10#1#',
            quality=0.47,
            cfg=cfg
        )
        cls.mark[1].update(['0', '2'])
        cls.mark[2].update(['1'])
        cls.mark[5].update(['0', '1'])
        cls.mark[7].update(['1'])

        # when
        new_cls = expected_case(cls, p0, time)

        # then
        assert new_cls is not None
        # One `random` attribute gets specified
        assert 5 == new_cls.condition.specificity
        assert Effect('0##10#1#') == new_cls.effect
        assert 7 == new_cls.action
        assert new_cls.mark.is_empty() is True
        assert 0.5 == new_cls.q

    def test_should_handle_unexpected_case_1(self, cfg):
        # given
        cls = Classifier(action=2, cfg=cfg)

        p0 = Perception('01100000')
        p1 = Perception('10100010')
        time = 14

        new_cls = unexpected_case(cls, p0, p1, time)

        # Quality should be decreased
        assert 0.475 == cls.q

        # Should be marked with previous perception
        for mark_attrib in cls.mark:
            assert 1 == len(mark_attrib)

        assert '0' in cls.mark[0]
        assert '1' in cls.mark[1]
        assert '1' in cls.mark[2]
        assert '0' in cls.mark[3]
        assert '0' in cls.mark[4]
        assert '0' in cls.mark[5]
        assert '0' in cls.mark[6]
        assert '0' in cls.mark[7]

        # New classifier should not be the same object
        assert cls is not new_cls

        # Check attributes of a new classifier
        assert Condition('01####0#') == new_cls.condition
        assert 2 == new_cls.action
        assert Effect('10####1#') == new_cls.effect

        # There should be no mark
        for mark_attrib in new_cls.mark:
            assert 0 == len(mark_attrib)

        assert 0.5 == new_cls.q
        assert cls.r == new_cls.r
        assert time == new_cls.tga
        assert time == new_cls.talp

    def test_should_handle_unexpected_case_2(self, cfg):
        # given
        cls = Classifier(
            condition='#######0',
            action=4,
            quality=0.4,
            cfg=cfg)
        cls.mark[0].update([0, 1])
        cls.mark[1].update([0, 1])
        cls.mark[2].update([0, 1])
        cls.mark[3].update([0, 1])
        cls.mark[4].update([1])
        cls.mark[5].update([0, 1])
        cls.mark[6].update([0, 1])

        p0 = Perception('11101010')
        p1 = Perception('10011101')
        time = 94

        # when
        new_cl = unexpected_case(cls, p0, p1, time)

        # then
        assert new_cl.condition == Condition('#110#010')
        assert new_cl.effect == Effect('#001#101')
        assert new_cl.mark.is_empty() is True
        assert time == new_cl.tga
        assert time == new_cl.talp
        assert abs(cls.q - 0.38) < 0.01

    def test_should_handle_unexpected_case_3(self, cfg):
        cls = Classifier(
            condition='#####1#0',
            effect='#####0#1',
            quality=0.475,
            cfg=cfg
        )

        cls.mark[0] = '1'
        cls.mark[1] = '1'
        cls.mark[2] = '0'
        cls.mark[3] = '1'
        cls.mark[5] = '1'
        cls.mark[7] = '1'

        p0 = Perception('11001110')
        p1 = Perception('01110000')
        time = 20

        new_cls = unexpected_case(cls, p0, p1, time)

        # Quality should be decreased
        assert 0.45125 == cls.q

        # No classifier should be generated here
        assert new_cls is None

    def test_should_handle_unexpected_case_5(self, cfg):
        # given
        cls = Classifier(
            condition='00####1#',
            action=2,
            effect='########',
            quality=0.129,
            reward=341.967,
            intermediate_reward=130.369,
            experience=201,
            tga=129,
            talp=9628,
            tav=25.08,
            cfg=cfg
        )
        cls.mark[2] = '2'
        cls.mark[3] = '1'
        cls.mark[4] = '1'
        cls.mark[5] = '0'
        cls.mark[7] = '0'

        p0 = Perception('00211010')
        p1 = Perception('00001110')
        time = 9628

        # when
        new_cls = unexpected_case(cls, p0, p1, time)

        # then
        assert new_cls is not None
        assert Condition('0021#01#') == new_cls.condition
        assert Effect('##00#1##') == new_cls.effect
        assert abs(0.5 - new_cls.q) < 0.1
        assert abs(341.967 - new_cls.r) < 0.1
        assert abs(130.369 - new_cls.ir) < 0.1
        assert abs(25.08 - new_cls.tav) < 0.1
        assert 1 == new_cls.exp
        assert 1 == new_cls.num
        assert time == new_cls.tga
        assert time == new_cls.talp

    def test_should_handle_unexpected_case_6(self, cfg):
        # given
        cls = Classifier(
            condition='0#1####1',
            action=2,
            effect='1#0####0',
            quality=0.38505,
            reward=1.20898,
            intermediate_reward=0,
            experience=11,
            tga=95,
            talp=873,
            tav=71.3967,
            cfg=cfg
        )
        cls.mark[1].update(['1'])
        cls.mark[3].update(['1'])
        cls.mark[4].update(['0', '1'])
        cls.mark[5].update(['1'])
        cls.mark[6].update(['0', '1'])

        p0 = Perception('01111101')
        p1 = Perception('11011110')
        time = 873

        # when
        new_cls = unexpected_case(cls, p0, p1, time)

        # then
        assert new_cls is not None
        assert Condition('0#1###01') == new_cls.condition
        assert Effect('1#0###10') == new_cls.effect
        assert abs(0.5 - new_cls.q) < 0.1
        assert abs(1.20898 - new_cls.r) < 0.1
        assert abs(0 - new_cls.ir) < 0.1
        assert abs(71.3967 - new_cls.tav) < 0.1
        assert 1 == new_cls.exp
        assert 1 == new_cls.num
        assert time == new_cls.tga
        assert time == new_cls.talp

    def test_should_create_new_classifier_using_covering(self, cfg):
        # given
        action_no = 2
        time = 123
        p0 = Perception('01001101')
        p1 = Perception('00011111')

        # when
        new_cl = cover(p0, action_no, p1, time, cfg)

        # then
        assert Condition('#1#0##0#') == new_cl.condition
        assert 2 == new_cl.action
        assert Effect('#0#1##1#') == new_cl.effect
        assert 0.5 == new_cl.q
        assert 0.5 == new_cl.r
        assert 0 == new_cl.ir
        assert 0 == new_cl.tav
        assert time == new_cl.tga
        assert time == new_cl.talp
        assert 1 == new_cl.num
        assert 1 == new_cl.exp
