import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer
import pytest


class TestRealMultiplexer:

    @pytest.fixture
    def rmpx(self):
        return gym.make('real-multiplexer-3bit-v0')

    def test_should_initialize_multiplexer_environment(self, rmpx):
        assert rmpx is not None
