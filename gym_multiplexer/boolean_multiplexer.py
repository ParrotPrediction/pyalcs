import logging
import random

import gym
from bitstring import BitArray
from gym.spaces import Discrete


class BooleanMultiplexer(gym.Env):

    REWARD = 1000

    def __init__(self, control_bits=3) -> None:
        self.control_bits = control_bits
        self.metadata = {'render.modes': ['human']}
        self.observation_space = Discrete(self._observation_string_length)
        self.action_space = Discrete(2)

    def _reset(self):
        logging.debug("Resetting the environment")
        bits = BitArray([random.randint(0, 1) for _ in
                         range(0, self._observation_string_length)])
        bits[-1] = False  # set validation bit to False

        self._ctrl_bits = bits[:self.control_bits]
        self._data_bits = bits[self.control_bits:]
        self._validation_bit = bits[-1]

        return self._observation()

    def _step(self, action):
        state = self._observation()
        reward = 0

        if action == self._answer:
            state = state[:-1] + '1'  # set validation bit to True
            reward = self.REWARD

        return state, reward, None, None

    def _render(self, mode='human', close=False):
        if close:
            return

        if mode == 'human':
            return self._observation()
        else:
            super(BooleanMultiplexer, self).render(mode=mode)

    def _observation(self) -> str:
        return (self._ctrl_bits + self._data_bits + self._validation_bit).bin

    @property
    def _observation_string_length(self):
        return self.control_bits + pow(2, self.control_bits) + 1

    @property
    def _answer(self):
        return int(self._data_bits[self._ctrl_bits.uint])
