import random

from gym.spaces import Discrete

from .multiplexer import Multiplexer


class BooleanMultiplexer(Multiplexer):

    def __init__(self, control_bits=3) -> None:
        super().__init__(control_bits)
        self.observation_space = Discrete(self._observation_string_length)
        self.action_space = Discrete(2)

    def _generate_state(self):
        return [random.randint(0, 1) for _ in range(0, self._observation_string_length - 1)]

    def _internal_state(self):
      return list(map(lambda x: round(x), self._state))
