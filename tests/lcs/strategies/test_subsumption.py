from dataclasses import dataclass

import pytest

import lcs.agents.acs2 as acs2
import lcs.agents.racs as racs
from lcs.agents.racs import Effect, Condition
from lcs.representations import Interval
from lcs.strategies.subsumption import find_subsumers, \
    is_subsumer, does_subsume


@dataclass
class SimpleRACSClassifier:
    condition: Condition
    effect: Effect
    action: int = 0
    exp: int = 0

    def is_reliable(self):
        pass

    def is_marked(self):
        pass

    def is_more_general(self):
        pass


class TestSubsumption:

    @pytest.fixture
    def acs2_cfg(self):
        return acs2.Configuration(8, 8)

    @pytest.fixture
    def racs_cfg(self):
        return racs.Configuration(classifier_length=2,
                                  number_of_possible_actions=2)

    def test_should_find_subsumer(self, acs2_cfg):
        # given
        subsumer = acs2.Classifier(
            condition='###0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=acs2_cfg)

        nonsubsumer = acs2.Classifier(action=3, cfg=acs2_cfg)

        population = acs2.ClassifiersList(
            *[nonsubsumer, subsumer, nonsubsumer])

        cl = acs2.Classifier(
            condition='1##0####',
            action=3,
            effect='##1#####',
            quality=0.5,
            reward=0.35,
            experience=1,
            cfg=acs2_cfg)

        # when
        actual_subsumers = find_subsumers(cl, population, acs2_cfg.theta_exp)

        # then
        assert len(actual_subsumers) == 1
        assert actual_subsumers[0] == subsumer

    def test_should_find_subsumer_among_nonsubsumers(self, acs2_cfg):
        # given
        subsumer = acs2.Classifier(
            condition='###0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=acs2_cfg)

        nonsubsumer = acs2.Classifier(action=3, cfg=acs2_cfg)

        cl = acs2.Classifier(
            condition='1##0####',
            action=3,
            effect='##1#####',
            quality=0.5,
            reward=0.35,
            experience=1,
            cfg=acs2_cfg)

        population = acs2.ClassifiersList(
            *[nonsubsumer, subsumer, nonsubsumer])

        # when
        subsumers = find_subsumers(cl, population, acs2_cfg.theta_exp)

        # then
        assert len(subsumers) == 1
        assert subsumers[0] == subsumer

    def test_should_sort_more_general_subsumer_1(self, acs2_cfg):
        # given
        subsumer1 = acs2.Classifier(
            condition='1##0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=acs2_cfg)

        subsumer2 = acs2.Classifier(
            condition='###0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=acs2_cfg)

        nonsubsumer = acs2.Classifier(action=3, cfg=acs2_cfg)

        cl = acs2.Classifier(
            condition='11#0####',
            action=3,
            effect='##1#####',
            quality=0.5,
            reward=0.35,
            experience=1,
            cfg=acs2_cfg)

        population = acs2.ClassifiersList(
            *[nonsubsumer, subsumer2, subsumer1, nonsubsumer])

        # when
        subsumers = find_subsumers(cl, population, acs2_cfg.theta_exp)

        # then
        assert len(subsumers) == 2
        assert subsumers[0] == subsumer2
        assert subsumers[1] == subsumer1

    def test_should_sort_more_general_subsumer_2(self, acs2_cfg):
        # given
        subsumer1 = acs2.Classifier(
            condition='1##0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=acs2_cfg)

        subsumer2 = acs2.Classifier(
            condition='###0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=acs2_cfg)

        nonsubsumer = acs2.Classifier(action=3, cfg=acs2_cfg)

        cl = acs2.Classifier(
            condition='11#0####',
            action=3,
            effect='##1#####',
            quality=0.5,
            reward=0.35,
            experience=1,
            cfg=acs2_cfg)

        population = acs2.ClassifiersList(
            *[nonsubsumer, subsumer1, subsumer2, nonsubsumer])

        # when
        subsumers = find_subsumers(cl, population, acs2_cfg.theta_exp)

        # then
        assert len(subsumers) == 2
        assert subsumers[0] == subsumer2
        assert subsumers[1] == subsumer1

    def test_should_find_most_general_subsumer(self, acs2_cfg):
        # given
        subsumer1 = acs2.Classifier(
            condition='1##0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=acs2_cfg)

        subsumer2 = acs2.Classifier(
            condition='#1#0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35, experience=23,
            cfg=acs2_cfg)

        most_general = acs2.Classifier(
            condition='###0####',
            action=3,
            effect='##1#####',
            quality=0.93,
            reward=1.35,
            experience=23,
            cfg=acs2_cfg)

        nonsubsumer = acs2.Classifier(cfg=acs2_cfg)

        cl = acs2.Classifier(
            condition='11#0####',
            action=3,
            effect='##1#####',
            quality=0.5,
            reward=0.35,
            experience=1,
            cfg=acs2_cfg)

        population = acs2.ClassifiersList(
            *[nonsubsumer, subsumer1, nonsubsumer, most_general,
              subsumer2, nonsubsumer])

        # when
        subsumers = find_subsumers(cl, population, acs2_cfg.theta_exp)

        # then
        assert subsumers[0] == most_general

    def test_should_randomly_select_one_of_equally_general_subsumers(
            self, acs2_cfg):

        # given
        subsumer1 = acs2.Classifier(condition='1##0####',
                                    action=3,
                                    effect='##1#####',
                                    quality=0.93,
                                    reward=1.35,
                                    experience=23,
                                    cfg=acs2_cfg)

        subsumer2 = acs2.Classifier(condition='#1#0####',
                                    action=3,
                                    effect='##1#####',
                                    quality=0.93,
                                    reward=1.35,
                                    experience=23,
                                    cfg=acs2_cfg)

        nonsubsumer = acs2.Classifier(cfg=acs2_cfg)

        cl = acs2.Classifier(condition='11#0####',
                             action=3,
                             effect='##1#####',
                             quality=0.5,
                             reward=0.35,
                             experience=1,
                             cfg=acs2_cfg)

        population = acs2.ClassifiersList(
            *[nonsubsumer, subsumer1, subsumer2, nonsubsumer])

        # when
        subsumers = find_subsumers(cl, population, acs2_cfg.theta_exp)

        # then
        assert subsumers[0] in [subsumer1, subsumer2]
        assert subsumers[1] in [subsumer1, subsumer2]

    @pytest.mark.parametrize("_exp, _q, _is_subsumer", [
        (1, .5, False),  # too young classifier
        (30, .92, True),  # enough experience and quality
        (15, .92, False),  # not experienced enough
    ])
    def test_should_distinguish_classifier_as_subsumer(
            self, _exp, _q, _is_subsumer, acs2_cfg):
        # given
        cl = acs2.Classifier(experience=_exp, quality=_q, cfg=acs2_cfg)

        # when & then
        # general classifier should not be considered as subsumer
        assert is_subsumer(cl, acs2_cfg.theta_exp) is _is_subsumer

    def test_should_not_distinguish_marked_classifier_as_subsumer(
            self, acs2_cfg):
        # given
        # Now check if the fact that classifier is marked will block
        # it from being considered as a subsumer
        cl = acs2.Classifier(experience=30, quality=0.92, cfg=acs2_cfg)
        cl.mark[3].add('1')

        # when & then
        assert is_subsumer(cl, acs2_cfg.theta_exp) is False

    @pytest.mark.parametrize(
        "_cl1c, _cl1a, _cl1e, _cl1q, _cl1r, _cl1exp,"
        "_cl2c, _cl2a, _cl2e, _cl2q, _cl2r, _cl2exp,"
        "_result", [
            ("###0####", 3, '##1#####', 0.93, 1.35, 23,
             "1##0####", 3, '##1#####', 0.5, 0.35, 1,
             True),
            ("10##0#1#", 6, '01####0#', 0.84, 0.33, 3,
             "10####2#", 3, '01####0#', 0.5, 0.41, 1,
             False),
            ("######0#", 6, '########', 0.99, 11.4, 32,
             "###1##0#", 6, '########', 0.5, 9.89, 1,
             True)
        ])
    def test_should_subsume_another_classifier(
        self, _cl1c, _cl1a, _cl1e, _cl1q, _cl1r, _cl1exp,
        _cl2c, _cl2a, _cl2e, _cl2q, _cl2r, _cl2exp,
            _result, acs2_cfg):

        # given
        cl = acs2.Classifier(condition=_cl1c, action=_cl1a, effect=_cl1e,
                             quality=_cl1q, reward=_cl1r,
                             experience=_cl1exp, cfg=acs2_cfg)

        other = acs2.Classifier(condition=_cl2c, action=_cl2a, effect=_cl2e,
                                quality=_cl2q, reward=_cl2r,
                                experience=_cl2exp, cfg=acs2_cfg)

        # when & then
        assert does_subsume(cl, other, acs2_cfg.theta_exp) is _result

    @pytest.mark.parametrize(
        "_e1, _e2, _exp1, _marked, _reliable,"
        "_more_general, _condition_matching, _result", [
            ([Interval(.2, .4), Interval(.5, .6)],
             [Interval(.2, .4), Interval(.5, .6)],
             30, False, True, True, True, True),  # all good
            ([Interval(.2, .4), Interval(.5, .6)],
             [Interval(.2, .4), Interval(.5, .6)],
             30, False, False, True, True, False),  # not reliable
            ([Interval(.2, .4), Interval(.5, .6)],
             [Interval(.2, .4), Interval(.5, .6)],
             30, True, True, True, True, False),  # marked
            ([Interval(.2, .4), Interval(.5, .6)],
             [Interval(.2, .4), Interval(.5, .6)],
             30, False, True, False, True, False),  # less general
            ([Interval(.2, .4), Interval(.5, .6)],
             [Interval(.2, .4), Interval(.5, .6)],
             30, False, True, True, False, False),  # condition not matching
            ([Interval(.2, .4), Interval(.5, .6)],
             [Interval(.2, .4), Interval(.5, .7)],
             30, False, True, True, True, False),  # different effects
            ([Interval(.2, .4), Interval(.5, .6)],
             [Interval(.2, .4), Interval(.5, .6)],
             10, False, True, True, True, False),  # not experienced
        ])
    def test_should_detect_subsumption(self, _e1, _e2, _exp1, _marked,
                                       _reliable, _more_general,
                                       _condition_matching, _result,
                                       mocker, racs_cfg):
        # given
        cl1 = SimpleRACSClassifier(condition=racs.Condition.generic(racs_cfg),
                                   effect=racs.Effect(_e1, racs_cfg),
                                   exp=_exp1)
        cl2 = SimpleRACSClassifier(condition=racs.Condition.generic(racs_cfg),
                                   effect=racs.Effect(_e2, racs_cfg))

        # when
        mocker.patch.object(cl1, "is_reliable")
        mocker.patch.object(cl1, "is_marked")
        mocker.patch.object(cl1, "is_more_general")
        mocker.patch.object(cl1.condition, "subsumes")

        cl1.is_reliable.return_value = _reliable
        cl1.is_marked.return_value = _marked
        cl1.is_more_general.return_value = _more_general
        cl1.condition.subsumes.return_value = _condition_matching

        # then
        assert does_subsume(cl1, cl2, racs_cfg.theta_exp) == _result
