import gym
# noinspection PyUnresolvedReferences
import gym_maze
import pytest

from examples.acs2.maze.utils import maze_knowledge
from lcs.agents.acs2 import ACS2, Configuration
from .utils import count_microclassifiers, \
    count_macroclassifiers, \
    count_reliable


class TestMaze:

    @pytest.fixture
    def env(self):
        return gym.make('Woods1-v0')

    def test_should_traverse(self, env):
        # given
        cfg = Configuration(8, 8,
                            epsilon=1.0,
                            do_ga=False,
                            performance_fcn=maze_knowledge)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 300)

        # then
        assert 90 < count_macroclassifiers(population) < 200

        assert 100 == self._get_knowledge(metrics)

        assert count_macroclassifiers(population) == count_reliable(population)

        assert count_macroclassifiers(population) \
            == count_microclassifiers(population)

        assert self._get_total_steps(metrics) > 5000

    def test_should_traverse_with_ga(self, env):
        # given
        cfg = Configuration(8, 8,
                            epsilon=0.8,
                            mu=0.3,
                            chi=0.0,
                            do_ga=True,
                            performance_fcn=maze_knowledge)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 300)

        # then
        assert abs(380 - count_macroclassifiers(population)) < 55
        assert abs(100 - self._get_knowledge(metrics)) < 5

        assert count_macroclassifiers(population) \
            > count_reliable(population)

        assert count_macroclassifiers(population) \
            < count_microclassifiers(population)

        assert self._get_total_steps(metrics) > 2500

    @pytest.mark.skip(reason="implement it")
    def test_should_exploit_maze(self):
        pass

    @staticmethod
    def _get_knowledge(metrics):
        return metrics[-1]['performance']['knowledge']

    @staticmethod
    def _get_total_steps(metrics):
        return metrics[-1]['agent']['total_steps']
