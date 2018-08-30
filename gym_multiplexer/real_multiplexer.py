import random

from gym.spaces import Box, Discrete

from .multiplexer import Multiplexer


class RealMultiplexer(Multiplexer):

  def __init__(self, control_bits=3, threshold=.5) -> None:
    super().__init__(control_bits)
    self.threshold = threshold
    self.observation_space = Box(low=0, high=1, shape=(self._observation_string_length, ))
    self.action_space = Discrete(2)

  def _generate_state(self):
    return [random.random() for _ in range(0, self._observation_string_length - 1)]

  def _internal_state(self):
    return list(map(lambda x: x > self.threshold, self._state))
