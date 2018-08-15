import random
from typing import List

from lcs import Perception, TypedList
from lcs.agents.racs import Configuration, Condition
from lcs.representations import UBR


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

    def get_differences(self, p0: Perception) -> Condition:
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
        p0: Perception

        Returns
        -------
        Condition
            differences between mark and perception that can form
            a new classifier

        """
        diff = Condition.generic(self.cfg)

        if self.is_marked():
            enc_p0 = list(map(self.cfg.encoder.encode, p0))

            unique_diff_indices = \
                [pi for pi, p in enumerate(enc_p0) if p not in self[pi]]
            fuzzy_diff_indices = \
                [pi for pi, p in enumerate(enc_p0) if len(self[pi]) > 1]

            if len(unique_diff_indices) > 0:
                ridx = random.choice(unique_diff_indices)
                p = enc_p0[ridx]

                diff[ridx] = UBR(p, p)
            elif len(fuzzy_diff_indices) > 0:
                for pi, p in enumerate(enc_p0):
                    if len(self[pi]) > 1:
                        diff[pi] = UBR(p, p)

        return diff
