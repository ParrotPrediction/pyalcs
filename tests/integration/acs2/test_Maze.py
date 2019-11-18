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
                            biased_exploration=0.5,
                            do_ga=False,
                            metrics_trial_frequency=1,
                            user_metrics_collector_fcn=self._maze_metrics)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 300)

        # then
        assert 90 < count_macroclassifiers(population) < 200

        assert self._get_knowledge(metrics) == 100

        assert count_macroclassifiers(population) == count_reliable(population)

        assert count_macroclassifiers(population) \
            == count_microclassifiers(population)

        assert self._get_total_steps(metrics) > 5000

    def test_should_traverse_with_ga(self, env):
        # given
        cfg = Configuration(8, 8,
                            epsilon=0.8,
                            biased_exploration=0.5,
                            mu=0.3,
                            chi=0.0,
                            do_ga=True,
                            metrics_trial_frequency=1,
                            user_metrics_collector_fcn=self._maze_metrics)
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
        return metrics[-1]['knowledge']

    @staticmethod
    def _get_total_steps(metrics):
        return sum(m['steps_in_trial'] for m in metrics)

    @staticmethod
    def _maze_metrics(population, environment):
        return {
            'knowledge': maze_knowledge(population, environment)
        }
