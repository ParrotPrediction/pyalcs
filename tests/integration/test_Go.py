import gym
import pytest

from examples.go.utils import moves_9x9, process_state, calculate_bw_ratio


class TestGo:

    @pytest.fixture
    def env(self):
        return gym.make('Go9x9-v0')

    def test_should_get_all_possible_moves(self):
        # given
        possible_moves = moves_9x9()

        # then
        assert len(possible_moves) == 81
        assert len(set(possible_moves)) == 81

    def test_should_flatten_state(self, env):
        # given
        state = env.reset()

        # when
        flatten = process_state(state)

        # then
        assert len(flatten) == 81

    def test_should_calculate_ratio(self, env):
        # given
        state = env.reset()

        # when & then
        assert calculate_bw_ratio(env) == 0.0

        # when & then
        _, _, _, _ = env.step(38)  # move to C8
        assert calculate_bw_ratio(env) == 1.0
