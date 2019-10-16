from __future__ import annotations

from copy import copy
from typing import Any, Tuple

from lcs import TypedList


class PerceptionString(TypedList):

    def __init__(self, observation, wildcard='#', oktypes=(str, dict)):
        # TODO ProbabilityEnhancedAttribute instead of dict in oktypes
        super().__init__(oktypes, *observation)
        assert type(wildcard) in self.oktypes
        self.wildcard = wildcard

    @classmethod
    def empty(cls,
              length: int,
              wildcard: Any ='#',
              oktypes: Tuple[Any, Any]=(str, dict)):
        """
        Creates a perception string composed from wildcard symbols.
        Note that in case that wildcard is an object is get's copied
        not to be accessed by reference.

        Parameters
        ----------
        length: int
            length of perception string
        wildcard: Any
            wildcard symbol
        oktypes: Tuple[Any]
            tuple of allowed classes to represent perception string

        Returns
        -------
        PerceptionString
            generic perception string
        """
        ps_str = [copy(wildcard) for _ in range(length)]
        return cls(ps_str, wildcard=wildcard, oktypes=oktypes)

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

    def __eq__(self, other):
        return self._items == other._items

    def __repr__(self):
        return ''.join(map(str, self))
