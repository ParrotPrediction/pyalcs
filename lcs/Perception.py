import collections.abc


class Perception(collections.abc.Sequence):
    """
    Represents current state of the environment at given time instance.
    By default each environment attribute is represented as `str` type.
    """

    __slots__ = ['_items', 'oktypes']

    def __init__(self, observation, oktypes=(str,)):
        for el in observation:
            assert type(el) in oktypes

        self._items = tuple(observation)

    @classmethod
    def empty(cls):
        return cls([], oktypes=(None,))

    def __hash__(self):
        return hash(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self):
        return ' '.join(map(str, self))

    def __eq__(self, other):
        for si, oi in zip(self, other):
            if si != oi:
                return False

        return True
