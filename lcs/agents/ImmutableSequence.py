from copy import copy


class ImmutableSequence:

    WILDCARD = '#'
    OK_TYPES = (str, dict)  # PEEs are stored in dict

    def __init__(self, observation):
        obs = tuple(observation)

        assert type(self.WILDCARD) in self.OK_TYPES
        assert all(isinstance(o, self.OK_TYPES) for o in obs)

        self._items = obs

    @classmethod
    def empty(cls, length: int):
        ps_str = [copy(cls.WILDCARD) for _ in range(length)]
        return cls(ps_str)

    def subsumes(self, other) -> bool:
        """
        Checks if given perception string subsumes other one.

        Parameters
        ----------
        other: PerceptionString
            other perception string

        Returns
        -------
        bool
            True if `other` is subsumed by `self`, False otherwise
        """
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        assert isinstance(value, self.OK_TYPES)
        lst = list(self._items)
        lst[index] = value

        self._items = tuple(lst)

    def __eq__(self, other):
        return self._items == other._items

    def __hash__(self):
        return hash(self._items)

    def __repr__(self):
        return ''.join(map(str, self._items))
