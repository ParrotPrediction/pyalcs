import lcs.agents.acs2 as acs2
from lcs.strategies.action_selection import EpsilonGreedy


class Configuration(acs2.Configuration):
    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

        # ER replay memory buffer size
        self.buffer_size = kwargs.get('buffer_size', 1000)

        # ER replay memory samples number
        self.samples_number = kwargs.get('samples_number', 3)

    def __str__(self) -> str:
        return str(vars(self))
