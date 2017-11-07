import pytest
import gym
import gym_maze

from alcs.acs2 import ACS2Configuration
from alcs.acs2.ACS2 import ACS2

from examples.maze import calculate_knowledge


class TestMaze:

    @pytest.fixture
    def env(self):
        return gym.make('Woods1-v0')

    def test_should_traverse(self, env):
        # given
        cfg = ACS2Configuration(8, 8, do_ga=False)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 300)

        # then
        assert 100 < len(population) < 200

        knowledge = calculate_knowledge(env, population)
        assert 100 == knowledge

        reliable_count = len([cl for cl in population if cl.is_reliable()])
        assert len(population) == reliable_count

        total_cls = sum(cl.num for cl in population)
        assert len(population) == total_cls

    def test_should_traverse_with_ga(self, env):
        # given
        cfg = ACS2Configuration(8, 8, do_ga=True)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 300)

        # then
        assert len(population) > 300  # should be ~300 (actually about 800)

        knowledge = calculate_knowledge(env, population)
        assert 100 == knowledge

        reliable_count = len([cl for cl in population if cl.is_reliable()])
        assert len(population) > reliable_count

        total_cls = sum(cl.num for cl in population)
        assert len(population) < total_cls  # total_cls should be ~2000
