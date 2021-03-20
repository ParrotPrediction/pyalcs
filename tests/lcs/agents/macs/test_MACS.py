import pytest

from lcs import Perception
from lcs.agents.macs.macs import Configuration, MACS


class TestMACS:

    @pytest.fixture
    def cfg(self):
        feature_vals = {'0', '1'}
        return Configuration(2, 2, feature_possible_values=[feature_vals] * 2)

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
