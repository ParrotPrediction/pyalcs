import pytest

from lcs import Perception
from lcs.agents.yacs.yacs import Configuration, YACS


class TestYACS:

    @pytest.fixture
    def cfg(self):
        return Configuration(2, 2, feature_possible_values=[2, 2])

    @pytest.fixture
    def agent(self, cfg):
        return YACS(cfg)

    def test_should_validate_if_perception_is_in_range(self, agent):
        with pytest.raises(AssertionError):
            agent.remember_situation(Perception('02'))

    def test_should_remember_perception(self, agent):
        # given
        assert len(agent.desirability_values) == 0

        # when & then
        agent.remember_situation(Perception('00'))
        assert len(agent.desirability_values) == 1

        agent.remember_situation(Perception('01'))
        assert len(agent.desirability_values) == 2

        agent.remember_situation(Perception('00'))
        assert len(agent.desirability_values) == 2
