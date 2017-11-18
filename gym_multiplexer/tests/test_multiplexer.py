import logging
import random
import sys

import gym

# noinspection PyUnresolvedReferences
import gym_multiplexer

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


class TestMultiplexer:

    def test_should_initialize_multiplexer(self):
        # when
        mp = gym.make('boolean-multiplexer-6bit-v0')

        # then
        assert mp is not None
        assert 6 == mp.observation_space.n
        assert 2 == mp.action_space.n

    def test_should_render_state(self):
        # given
        mp = gym.make('boolean-multiplexer-3bit-v0')
        mp.reset()

        # when
        state = mp.render()

        # then
        assert state is not None
        assert 3 == len(state)

    def test_should_execute_step(self):
        # given
        mp = gym.make('boolean-multiplexer-3bit-v0')
        mp.reset()
        action = self._random_action()

        # when
        state, reward, done, _ = mp.step(action)

        # then
        assert state is not None
        assert reward in [0, 1]
        assert done is True

    @staticmethod
    def _random_action():
        return random.sample([0, 1], 1)
