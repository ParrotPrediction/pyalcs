import gym
import gym_handeye
import pytest

from examples.acs2.handeye.utils import calculate_performance
from lcs.agents.acs2 import Configuration, ACS2


class TestHandEye:

    @pytest.fixture
    def env(self):
        return gym.make('HandEye3-v0')

    def test_initialize_environment(self, env):
        assert env is not None

    def test_should_gain_knowledge(self, env):
        # given
        cfg = Configuration(env.observation_space.n,
                            env.action_space.n,
                            epsilon=1.0,
                            do_ga=False,
                            do_action_planning=True,
                            action_planning_frequency=50,
                            performance_fcn=calculate_performance)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 10)

        # then
        assert metrics[-1]['performance']['knowledge'] > 0.0
        assert metrics[-1]['performance']['with_block'] > 0.0
        assert metrics[-1]['performance']['no_block'] > 0.0

    def test_should_evaluate_knowledge(self, env):
        # given
        cfg = Configuration(env.observation_space.n,
                            env.action_space.n,
                            epsilon=1.0,
                            do_ga=False,
                            do_action_planning=True,
                            action_planning_frequency=50,
                            performance_fcn=calculate_performance)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 10)

        # then
        for metric in metrics:
            assert metric['performance']['knowledge'] >= 0.0
            assert metric['performance']['with_block'] >= 0.0
            assert metric['performance']['no_block'] >= 0.0
            assert metric['performance']['knowledge'] <= 100.0
            assert metric['performance']['with_block'] <= 100.0
            assert metric['performance']['no_block'] <= 100.0
