import pytest
import gym
import gym_maze

from alcs.acs2 import ACS2Configuration
from alcs.acs2.ACS2 import ACS2


class TestMaze:

    @pytest.fixture
    def env(self):
        return gym.make('Woods1-v0')

    def test_should_traverse(self, env):
        # given
        cfg = ACS2Configuration(8, 8, do_ga=False)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 500)

        # then
        assert 100 < len(population) < 200
