from lcs.agents import ImmutableSequence


class Effect(ImmutableSequence):

    def __init__(self, observation):
        super().__init__(observation)

