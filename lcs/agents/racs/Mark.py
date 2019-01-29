import random
from typing import List

from lcs import Perception, TypedList, is_different
from lcs.agents.racs import Configuration, Condition
from lcs.representations import Interval


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

    def complement_marks(self, p: Perception) -> bool:
        """
        Adds current perception to the corresponding marking attributes.
        Only extends already marked attributes.

        Parameters
        ----------
        p: Perception
            agent's perception

        Returns
        -------
        bool
            True if any attribute was marked, False otherwise

        """
        changed = False

        for idx, mark in enumerate(self):
            new_elem = p[idx]

            if len(mark) > 0 and self._not_in_mark(new_elem, self[idx]):
                self[idx].add(new_elem)
                changed = True

        return changed

    def set_mark_using_condition(self,
                                 condition: Condition,
                                 p: Perception) -> bool:
        """
        Set's the mark using classifier condition.
        If it is already marked, it gets complemented using perception only.

        If it's not marked only "don't care" attributes get marked.

        Parameters
        ----------
        condition: Condition
            classifier condition
        p: Perception
            agent's perception

        Returns
        -------
        bool
            True if any attribute was marked, False otherwise
        """
        if self.is_marked():
            return self.complement_marks(p)

        changed = False

        for idx, item in enumerate(condition):
            if item == self.cfg.classifier_wildcard:
                self[idx].add(p[idx])
                changed = True

        return changed

    def get_differences(self, p: Perception) -> Condition:
        """
        Difference determination is run when the classifier anticipated the
        change correctly.

        If it's marked we want to find if we can propose differences that
        will be applied to new condition part (specialization).

        There can be two types of differences:
        1) unique - one or more attributes in mark does not contain given
        perception attribute
        2) fuzzy - there is no unique difference - one or more attributes in
        the mark specify more than one value in perception attribute.

        If only unique differences are present - one random one get specified.
        If there are fuzzy differences everyone is specified.

        Parameters
        ----------
        p: Perception

        Returns
        -------
        Condition
            differences between mark and perception that can form
            a new classifier

        """
        diff = Condition.generic(self.cfg)

        if self.is_marked():

            # Unique and fuzzy difference counts
            nr1, nr2 = 0, 0

            # Count difference types
            for idx, item in enumerate(self):
                if len(item) > 0 and self._not_in_mark(p[idx], item):
                    nr1 += 1
                elif len(item) > 1:
                    nr2 += 1

            if nr1 > 0:
                possible_idx = [p_idx for p_idx, p in enumerate(p) if
                                self._not_in_mark(p, self[p_idx]) and
                                len(self[p_idx]) > 0]

                rand_idx = random.choice(possible_idx)
                diff[rand_idx] = Interval(p[rand_idx], p[rand_idx])
            elif nr2 > 0:
                for pi, p_val in enumerate(p):
                    if len(self[pi]) > 1:
                        diff[pi] = Interval(p_val, p_val)

        return diff

    @staticmethod
    def _not_in_mark(val: float, mark):
        assert type(val) is float
        return all(is_different(val, attrib) for attrib in mark)
