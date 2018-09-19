import gym
import pytest

# noinspection PyUnresolvedReferences
import gym_maze
from lcs.agents.acs2 import ACS2, Configuration
from examples.acs2.maze.utils import calculate_performance
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
                            performance_fcn=calculate_performance)
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
                            performance_fcn=calculate_performance)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 1500)

        # then
        print(f"Macroclassifiers: {count_macroclassifiers(population)}")
        print(f"Microclassifiers: {count_microclassifiers(population)}")
        print(f"Reliable: {count_reliable(population)}")
        print(f"Knowledge: {self._get_knowledge(metrics)}")
        print(f"Total steps: {self._get_total_steps(metrics)}")

        reliable = [cl for cl in population if cl.is_reliable()]
        for cl in sorted(reliable, key=lambda cl: -cl.fitness):
            print(f"{cl} num: {cl.num} q: {cl.q:.2f} fitness: {cl.fitness:.2f}")

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
