from random import choice
from typing import Optional, List

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

    def get_differences(self, perception: Perception) -> Optional[Condition]:
        """
        Determines the strongest differences in between the mark
        and perception.

        :param: perception
        :return: condition that specifies all the differences.
        """
        nr1 = 0
        nr2 = 0

        # Count difference types
        for idx, item in enumerate(self):
            if len(item) > 0 and perception[idx] not in item:
                nr1 += 1
            elif len(item) > 1:
                nr2 += 1

        if nr1 > 0:
            # One or more absolute differences detected -> specialize one
            # randomly chosen
            condition = Condition.empty(
                wildcard=self.cfg.classifier_wildcard,
                length=self.cfg.classifier_length)

            possible_idx = []
            for idx, item in enumerate(self):
                if len(item) > 0 and perception[idx] not in item:
                    possible_idx.append(idx)

            rand_idx = choice(possible_idx)
            condition[rand_idx] = perception[rand_idx]
        elif nr2 > 0:
            # One or more equal differences detected -> specialize all of them
            condition = Condition.empty(
                wildcard=self.cfg.classifier_wildcard,
                length=self.cfg.classifier_length)

            for idx, item in enumerate(self):
                if len(item) > 1:
                    condition[idx] = perception[idx]
        else:
            # Nothing for specialization found
            return None

        return condition
