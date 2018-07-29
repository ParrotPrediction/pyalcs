import gym
import pytest

# noinspection PyUnresolvedReferences
import gym_maze
from lcs.agents.acs2 import ACS2, ACS2Configuration
from integration.maze.utils import calculate_performance
from .utils import count_microclassifiers, \
    count_macroclassifiers, \
    count_reliable


class TestMaze:

    @pytest.fixture
    def env(self):
        return gym.make('Woods1-v0')

    def test_should_traverse(self, env):
        # given
        cfg = ACS2Configuration(8, 8,
                                epsilon=1.0,
                                do_ga=False,
                                performance_fcn=calculate_performance)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 300)

        # then
        assert 100 < count_macroclassifiers(population) < 200

        assert 100 == self._get_knowledge(metrics)

        assert count_macroclassifiers(population) == count_reliable(population)

        assert count_macroclassifiers(population) \
            == count_microclassifiers(population)

        assert self._get_total_steps(metrics) > 5000

    def test_should_traverse_with_ga(self, env):
        # given
        cfg = ACS2Configuration(8, 8,
                                epsilon=1.0,
                                do_ga=True,
                                performance_fcn=calculate_performance)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 300)

        # then
        assert abs(250 - count_macroclassifiers(population)) < 55

        assert 100 == self._get_knowledge(metrics)

        assert count_macroclassifiers(population) \
            > count_reliable(population)

        assert count_macroclassifiers(population) \
            <= count_microclassifiers(population)

        assert self._get_total_steps(metrics) > 5000

    @pytest.mark.skip(reason="implement it")
    def test_should_exploit_maze(self):
        pass

    @staticmethod
    def _get_knowledge(metrics):
        return metrics[-1]['performance']['knowledge']

    @staticmethod
    def _get_total_steps(metrics):
        return metrics[-1]['agent']['total_steps']
