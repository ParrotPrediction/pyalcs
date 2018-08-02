from copy import copy

from lcs import TypedList


class PerceptionString(TypedList):

    def __init__(self, observation, wildcard='#', oktypes=(str,)):
        super().__init__(oktypes, *observation)
        self.wildcard = wildcard

    @classmethod
    def empty(cls, length, wildcard='#', oktypes=(str,)):
        ps_str = [copy(wildcard) for _ in range(length)]
        return cls(ps_str, wildcard=wildcard, oktypes=oktypes)

    def __repr__(self):
        return ''.join(map(str, self))
