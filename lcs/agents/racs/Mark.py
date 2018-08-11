from typing import List

from lcs import Perception, TypedList
from lcs.agents.racs import Configuration, Condition


class Mark(TypedList):

    def __init__(self, cfg: Configuration) -> None:
        self.cfg = cfg
        initial: List = [set() for _ in range(self.cfg.classifier_length)]
        super().__init__((set,), *initial)

    def is_marked(self) -> bool:
        """
        Returns
        -------
        bool
            If mark is specified at any attribute
        """
        return any(len(attrib) != 0 for attrib in self)

    def complement_marks(self, perception: Perception) -> bool:
        """
        Adds current perception to the corresponding marking attributes.
        Only extends already marked attributes.
        Perception get's encoded before being stored.

        Parameters
        ----------
        perception: Perception
            agent's perception

        Returns
        -------
        bool
            True if any attribute was marked, False otherwise

        """
        changed = False
        encoded_perception = list(map(self.cfg.encoder.encode, perception))

        for idx, attrib in enumerate(self):
            new_elem = encoded_perception[idx]
            if len(attrib) > 0 and new_elem not in self[idx]:
                self[idx].add(new_elem)
                changed = True

        return changed

    def set_mark_using_condition(self,
                                 condition: Condition,
                                 perception: Perception) -> bool:
        """
        Set's the mark using classifier condition.
        If it is already marked, it gets complemented using perception only.

        If it's not marked only "don't care" attributes get marked.

        Parameters
        ----------
        condition: Condition
            classifier condition
        perception: Perception
            agent's perception

        Returns
        -------
        bool
            True if any attribute was marked, False otherwise
        """
        if self.is_marked():
            return self.complement_marks(perception)

        changed = False
        encoded_perception = list(map(self.cfg.encoder.encode, perception))

        for idx, item in enumerate(condition):
            if item == self.cfg.classifier_wildcard:
                self[idx].add(encoded_perception[idx])
                changed = True

        return changed
