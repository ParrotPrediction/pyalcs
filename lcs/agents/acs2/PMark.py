import random
from typing import List

from lcs import Perception, TypedList
from . import Configuration, Condition


class PMark(TypedList):
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
        Directly further specializes all specified attributes in the mark

        :param perception:
        :return: True if something was specialized
        """
        changed = False

        for idx, item in enumerate(self):
            if len(item) > 0:
                self[idx].add(perception[idx])
                changed = True

        return changed

    def set_mark_using_condition(self,
                                 condition: Condition,
                                 perception: Perception) -> bool:
        if self.is_marked():
            # Mark is already specified. Further specialize all
            # specified attributes
            return self.complement_marks(perception)

        changed = False

        for idx, item in enumerate(condition):
            if item == self.cfg.classifier_wildcard:
                self[idx].add(perception[idx])
                changed = True

        return changed

    def get_differences(self, p0: Perception) -> Condition:
        """
        Determines the strongest differences in between the mark
        and current perception.

        :param: perception
        :return: condition that specifies all the differences.
        """
        diff = Condition.empty(
            wildcard=self.cfg.classifier_wildcard,
            length=self.cfg.classifier_length)

        nr1, nr2 = 0, 0

        # Count difference types
        for idx, item in enumerate(self):
            if len(item) > 0 and p0[idx] not in item:
                nr1 += 1
            elif len(item) > 1:
                nr2 += 1

        if nr1 > 0:
            possible_idx = [pi for pi, p in enumerate(p0) if
                            p not in self[pi] and len(self[pi]) > 0]
            rand_idx = random.choice(possible_idx)
            diff[rand_idx] = p0[rand_idx]
        elif nr2 > 0:
            for idx, item in enumerate(self):
                if len(item) > 1:
                    diff[idx] = p0[idx]

        return diff

    def __repr__(self):
        if self.is_marked():
            return "Marked"
        else:
            return "Empty"
