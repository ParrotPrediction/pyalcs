from gym.spaces import Discrete

from .multiplexer import Multiplexer


class BooleanMultiplexer(Multiplexer):

    def __init__(self, control_bits=3) -> None:
        super().__init__(control_bits)
        self.observation_space = Discrete(self._observation_string_length)
        self.action_space = Discrete(2)

    def _internal_state(self):
      return map(lambda x: round(x), self._state)
