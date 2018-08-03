import collections.abc

from . import check_types


class Perception(collections.abc.Sequence):
    """
    Represents current state of the environment at given time instance.
    By default each environment attribute is represented as `str` type.
    """
    def __init__(self, observation, oktypes=(str,)):
        self._items = list()

        for el in observation:
            check_types(oktypes, el)

        self._items.extend(list(observation))

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self):
        return ' '.join(map(str, self))
