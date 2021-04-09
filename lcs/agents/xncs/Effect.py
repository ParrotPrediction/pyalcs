from lcs.agents import ImmutableSequence


class Effect(ImmutableSequence):

    def __init__(self, observation):
        super().__init__(observation)

    def subsumes(self, other) -> bool:
        for ci, oi in zip(self, other):
            if ci != self.WILDCARD and oi != self.WILDCARD and ci != oi:
                return False
        return True
