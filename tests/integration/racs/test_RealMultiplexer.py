import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer
import pytest

from lcs.agents.racs import Configuration, RACS
from lcs.representations.RealValueEncoder import RealValueEncoder


class TestRealMultiplexer:

    @pytest.fixture
    def rmpx(self):
        return gym.make('real-multiplexer-3bit-v0')

    def test_should_initialize_multiplexer_environment(self, rmpx):
        assert rmpx is not None

    def test_should_execute_with_merging(self, rmpx):
        # given
        cfg = Configuration(rmpx.observation_space.shape[0],
                            rmpx.action_space.n,
                            encoder=RealValueEncoder(2),
                            do_ga=False,
                            do_merging=True)

        agent = RACS(cfg)

        # when
        population, metrics = agent.explore(rmpx, 100)

        # then
        for metric in metrics:
            assert metric['reward'] in {0, 1000}
