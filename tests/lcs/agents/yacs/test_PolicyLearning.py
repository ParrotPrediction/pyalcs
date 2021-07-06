import pytest

from lcs import Perception
from lcs.agents.yacs import Configuration
from lcs.agents.yacs.yacs import PolicyLearning, Classifier, ClassifiersList


class TestPolicyLearning:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=4,
                             number_of_possible_actions=4,
                             trace_length=5,
                             feature_possible_values=[{0, 1}] * 4)

    @pytest.fixture
    def pl(self, cfg):
        return PolicyLearning(cfg)

    def test_should_select_action(self, pl, cfg):
        # given
        cl1 = Classifier(condition='1###', action=0, effect='##1#',
                         reward=0.2, cfg=cfg)
        cl2 = Classifier(condition='1###', action=1, effect='#1#0',
                         reward=0.3, cfg=cfg)
        cl3 = Classifier(condition='1###', action=2, effect='####',
                         reward=0.1, cfg=cfg)

        match_set = ClassifiersList(*[cl1, cl2, cl3])
        p = Perception('1000')

        # when
        action = pl.select_action(match_set, {}, p)

        # then
        assert action == 1

    def test_should_return_random_action_with_empty_match_set(self, pl, cfg):
        # given
        match_set = ClassifiersList()
        p = Perception('1000')

        # when
        action = pl.select_action(match_set, {}, p)

        # then
        assert action in range(0, cfg.number_of_possible_actions)
