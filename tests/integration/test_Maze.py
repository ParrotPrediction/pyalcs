import gym
import pytest

# noinspection PyUnresolvedReferences
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
        cfg = ACS2Configuration(8, 8, epsilon=1.0, do_ga=False)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 300)

        # then
        assert 100 < self._count_macroclassifiers(population) < 200

        assert 100 == self._get_knowledge(env, population)

        assert self._count_macroclassifiers(population) \
            == self._count_reliable(population)

        assert self._count_macroclassifiers(population)\
            == self._count_microclassifiers(population)

        assert self._get_total_steps(metrics) > 5000

    def test_should_traverse_with_ga(self, env):
        # given
        cfg = ACS2Configuration(8, 8, epsilon=1.0, do_ga=True)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 300)

        # then
        # should be ~300 (actually about 800)
        assert self._count_macroclassifiers(population) > 300

        assert 100 == self._get_knowledge(env, population)

        assert self._count_macroclassifiers(population) \
            > self._count_reliable(population)

        # total_cls should be ~2000
        assert self._count_macroclassifiers(population) \
            < self._count_microclassifiers(population)

        assert self._get_total_steps(metrics) > 5000

    @pytest.mark.skip(reason="implement it")
    def test_should_exploit_maze(self):
        pass

    @staticmethod
    def _count_macroclassifiers(population):
        return len(population)

    @staticmethod
    def _count_microclassifiers(population):
        return sum(cl.num for cl in population)

    @staticmethod
    def _count_reliable(population):
        return len([cl for cl in population if cl.is_reliable()])

    @staticmethod
    def _get_knowledge(env, population):
        return calculate_knowledge(env, population)

    @staticmethod
    def _get_total_steps(metrics):
        return metrics[-1]['agent']['total_steps']
