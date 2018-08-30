import random

import gym

from .utils import get_correct_answer


class Multiplexer(gym.Env):

  REWARD = 1000

  def _generate_state(self): raise NotImplementedError
  def _internal_state(self): raise NotImplementedError

  def __init__(self, control_bits=3) -> None:
    self.control_bits = control_bits
    self.metadata = {'render.modes': ['human']}

    self._state = None
    self._validation_bit = 0

  def _reset(self):
      self._state = self._generate_state()
      self._validation_bit = 0
      return self._observation

  def _step(self, action):
      reward = 0

      if action == self._correct_answer:
          self._validation_bit = 1
          reward = self.REWARD

      return self._observation, reward, None, None

  def _render(self, mode='human', close=False):
      if close:
          return

      if mode == 'human':
          return self._observation

      return self.render(mode=mode)

  @property
  def _observation(self) -> list:
    observation = list(self._state)
    observation.append(self._validation_bit)
    return observation

  @property
  def _correct_answer(self):
    return get_correct_answer(list(self._internal_state()) , self.control_bits)

  @property
  def _observation_string_length(self):
    return self.control_bits + pow(2, self.control_bits) + 1