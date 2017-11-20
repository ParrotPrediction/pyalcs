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
        assert 7 == mp.observation_space.n
        assert 2 == mp.action_space.n

    def test_should_return_observation_when_reset(self):
        # given
        mp = gym.make('boolean-multiplexer-6bit-v0')

        # when
        state = mp.reset()

        # then
        assert state is not None
        assert state[-1] == '0'
        assert 7 == len(state)
        assert type(state) is str

    def test_should_render_state(self):
        # given
        mp = gym.make('boolean-multiplexer-3bit-v0')
        mp.reset()

        # when
        state = mp.render()

        # then
        assert state is not None
        assert state[-1] == '0'
        assert 4 == len(state)
        assert type(state) is str

    def test_should_execute_step(self):
        # given
        mp = gym.make('boolean-multiplexer-3bit-v0')
        mp.reset()
        action = self._random_action()

        # when
        state, reward, done, _ = mp.step(action)

        # then
        assert state is not None
        assert state[-1] in ['0', '1']
        assert type(state) is str
        assert reward in [0, 1000]
        assert done is True

    @staticmethod
    def _random_action():
        return random.sample([0, 1], 1)[0]
