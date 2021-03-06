import collections.abc
from typing import Generic, TypeVar, List, Iterator

from . import check_types

T = TypeVar('T')


class TypedList(collections.abc.MutableSequence, Generic[T]):
    __slots__ = ['_items', 'oktypes']

    def __init__(self, *args, oktypes):
        self._items: List[T] = list()
        self.oktypes = oktypes

        for el in args:
            check_types(self.oktypes, el)

        self._items.extend(list(args))

    def insert(self, index: int, o: T) -> None:
        check_types(self.oktypes, o)
        self._items.insert(index, o)

    def safe_remove(self, o: T) -> None:
        try:
            self.remove(o)
        except ValueError:
            pass

    def sort(self, *args, **kwargs) -> None:
        self._items.sort(*args, **kwargs)

    def __iter__(self) -> Iterator[T]:
        return super().__iter__()

    def __repr__(self):
        return f"{len(self._items)} items"

    def __setitem__(self, i: int, o: T):
        check_types(self.oktypes, o)
        self._items[i] = o

    def __delitem__(self, i):
        del self._items[i]

    def __getitem__(self, i) -> T:
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self):
        return hash((self.oktypes, self._items))

    def __eq__(self, o) -> bool:
        return self.oktypes == o.oktypes \
               and self._items == o._items
