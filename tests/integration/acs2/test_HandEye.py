import gym
# noinspection PyUnresolvedReferences
import gym_handeye
import pytest

from examples.acs2.handeye.utils import handeye_metrics
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
                            metrics_trial_frequency=1,
                            user_metrics_collector_fcn=handeye_metrics)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 20)

        # then
        assert metrics[-1]['knowledge'] > 0.0
        assert metrics[-1]['with_block'] > 0.0
        assert metrics[-1]['no_block'] > 0.0

    def test_should_evaluate_knowledge(self, env):
        # given
        cfg = Configuration(env.observation_space.n,
                            env.action_space.n,
                            epsilon=1.0,
                            do_ga=False,
                            do_action_planning=True,
                            action_planning_frequency=50,
                            user_metrics_collector_fcn=handeye_metrics)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(env, 10)

        # then
        for metric in metrics:
            assert 0.0 <= metric['knowledge'] <= 100.0
            assert 0.0 <= metric['with_block'] <= 100.0
            assert 0.0 <= metric['no_block'] <= 100.0
