import pytest

from lcs import Perception
from lcs.agents.macs.macs import Configuration, MACS, Classifier, ClassifiersList


class TestMACS:

    @pytest.fixture
    def cfg(self):
        feature_vals = {'0', '1'}
        return Configuration(2, 2, feature_possible_values=[feature_vals] * 2)

    @pytest.fixture
    def cfg2(self):
        feature_vals = {'0', '1'}
        return Configuration(8, 2, feature_possible_values=[feature_vals] * 8)

    @pytest.fixture
    def agent(self, cfg):
        return MACS(cfg)

    def test_should_validate_if_perception_is_in_range(self, agent):
        with pytest.raises(AssertionError):
            agent.remember_situation(Perception('02'))

    def test_should_remember_perception(self, agent):
        # given
        assert len(agent.desirability_values) == 0

        # when & then
        p0 = Perception('00')
        agent.remember_situation(p0)
        assert len(agent.desirability_values) == 1
        assert p0 in agent.desirability_values

        p1 = Perception('01')
        agent.remember_situation(p1)
        assert len(agent.desirability_values) == 2
        assert p1 in agent.desirability_values

        p2 = Perception('00')
        agent.remember_situation(p2)
        assert len(agent.desirability_values) == 2

    def test_should_get_anticipations(self, cfg2):
        # given
        p0 = Perception('11001100')
        action = 0
        pop = ClassifiersList(*[
            Classifier(condition='1#######', action=0, effect='??????1?', cfg=cfg2),
            Classifier(condition='#1######', action=0, effect='???????1', cfg=cfg2),
            Classifier(condition='##0#####', action=0, effect='0???????', cfg=cfg2),
            Classifier(condition='###0####', action=0, effect='?0??????', cfg=cfg2),
            Classifier(condition='####1###', action=0, effect='??1?????', cfg=cfg2),
            Classifier(condition='#####1##', action=0, effect='???1????', cfg=cfg2),
            Classifier(condition='######0#', action=0, effect='????0???', cfg=cfg2),
            Classifier(condition='#######0', action=0, effect='?????0??', cfg=cfg2),
            Classifier(condition='#######0', action=0, effect='?????1??', cfg=cfg2),
            # Invalid action
            Classifier(condition='1#######', action=1, effect='?????1??', cfg=cfg2),
            # Invalid condition
            Classifier(condition='0#######', action=0, effect='?????1??', cfg=cfg2),
        ])

        # when
        agent = MACS(population=pop, cfg=cfg2)
        anticipations = list(agent.get_anticipations(p0, action))

        # then
        assert len(anticipations) == 2
        assert anticipations == [
            Perception('00110011'),
            Perception('00110111')
        ]

