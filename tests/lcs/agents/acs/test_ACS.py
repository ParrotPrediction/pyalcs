import pytest

from lcs.agents.acs import Configuration, Condition, Effect
from lcs.agents.acs.ACS import ACS


class TestACS:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=4, number_of_possible_actions=4)

    def test_should_build_initial_population(self, cfg):
        agent = ACS(cfg)

        assert len(agent.population) == 4

        actions = [cl.action for cl in agent.population]
        assert actions == [0, 1, 2, 3]

        for cl in agent.population:
            assert cl.condition == Condition("####")
            assert cl.effect == Effect("####")

