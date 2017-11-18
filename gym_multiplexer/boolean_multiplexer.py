import logging
import random

import gym
from bitstring import BitArray
from gym.spaces import Discrete


class BooleanMultiplexer(gym.Env):

    def __init__(self, control_bits=3) -> None:
        self.metadata = {'render.modes': ['human']}
        self.control_bits = control_bits
        self.observation_space = Discrete(len(self._observation_string_length))
        self.action_space = Discrete(2)

    def _reset(self):
        logging.debug("Resetting the environment")
        bits = BitArray([random.randint(0, 1) for _ in
                         self._observation_string_length])

        self._ctrl_bits = bits[:self.control_bits]
        self._data_bits = bits[self.control_bits:]

    def _step(self, action):
        state = self._observation()
        reward = 0

        if action == self._answer:
            reward = 1

        return state, reward, None, None

    def _render(self, mode='human', close=False):
        if close:
            return

        if mode == 'human':
            return self.control_bits + self._data_bits
        else:
            super(BooleanMultiplexer, self).render(mode=mode)

    def _observation(self):
        return self.control_bits + self._data_bits

    @property
    def _observation_string_length(self):
        return range(0, self.control_bits + pow(2, self.control_bits))

    @property
    def _answer(self):
        return int(self._data_bits[self._ctrl_bits.uint])
