import pytest

import lcs.agents.acs as acs
import lcs.agents.acs2 as acs2
from lcs.strategies.action_selection import EpsilonGreedy


class TestEpsilonGreedy:

    @pytest.fixture
    def acs_cfg(self):
        return acs.Configuration(
            classifier_length=4,
            number_of_possible_actions=4,
            epsilon=0.5)

    @pytest.fixture
    def acs2_cfg(self):
        return acs2.Configuration(
            classifier_length=4,
            number_of_possible_actions=4,
            epsilon=0.5)

    def test_should_raise_error_when_epsilon_is_missing(self):
        with pytest.raises(KeyError):
            EpsilonGreedy(4)

    def test_should_assign_custom_epsilon(self):
        strategy = EpsilonGreedy(4, epsilon=0.9)
        assert strategy.epsilon == 0.9

    def test_should_work_with_acs(self, acs_cfg):
        # given
        c1 = acs.Classifier(
            condition='1##1', action=0, effect='0###',
            quality=0.571313, reward=7.67011,
            cfg=acs_cfg
        )
        c2 = acs.Classifier(
            condition='1##1', action=0, effect='0###',
            quality=0.571313, reward=6.67011,
            cfg=acs_cfg
        )
        population = acs.ClassifiersList(*[c1, c2])

        # when
        eg = EpsilonGreedy(acs_cfg.number_of_possible_actions,
                           epsilon=0.0)
        best_action = eg(population)

        # then
        assert best_action == 0

    def test_should_work_with_acs2(self, acs2_cfg):
        # given
        c1 = acs2.Classifier(
            condition='1##1', action=0, effect='0###',
            quality=0.571313, reward=7.67011,
            cfg=acs2_cfg
        )
        c2 = acs2.Classifier(
            condition='1##1', action=1, effect='0###',
            quality=0.581313, reward=7.67011,
            cfg=acs2_cfg
        )
        population = acs2.ClassifiersList(*[c1, c2])

        # when
        eg = EpsilonGreedy(acs2_cfg.number_of_possible_actions,
                           epsilon=0.0)

        best_action = eg(population)

        # then
        assert best_action == 1
