from random import choice

from alcs.acs2 import Condition

from alcs.acs2 import default_configuration
from alcs import Perception
from alcs.acs2.ACS2Configuration import ACS2Configuration


class PMark(list):
    def __init__(self, cfg: ACS2Configuration = default_configuration):
        self.cfg = default_configuration
        list.__init__(self, [set() for _ in range(cfg.classifier_length)])

    def __len__(self):
        return sum(1 for m in self if len(m) > 0)

    def __setitem__(self, idx, value):
        if not isinstance(value, str):
            raise TypeError('Mark should be composed of string objects')

        self[idx].add(value)

    def set_mark(self, perception: Perception) -> bool:
        """
        Directly further specializes all specified attributes in the mark

        :param perception:
        :return: True if something was specialized
        """
        changed = False

        for idx, item in enumerate(self):
            if len(item) > 0:
                self[idx] = perception[idx]
                changed = True

        return changed

    def set_mark_using_condition(self,
                                 condition: Condition,
                                 perception: Perception) -> bool:
        if not self.is_empty():
            # Mark is already specified. Further specialize all
            # specified attributes
            return self.set_mark(perception)

        changed = False

        for idx, item in enumerate(condition):
            if item == self.cfg.classifier_wildcard:
                self[idx] = perception[idx]
                changed = True

        return changed

    def is_empty(self) -> bool:
        """
        Check if there is any mark

        :return: True if is marked, False otherwise
        """
        return not len(self) > 0

    def get_differences(self, perception: Perception) -> Condition:
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
            condition = Condition()

            possible_idx = []
            for idx, item in enumerate(self):
                if len(item) > 0 and perception[idx] not in item:
                    possible_idx.append(idx)

            rand_idx = choice(possible_idx)
            condition[rand_idx] = perception[rand_idx]
        elif nr2 > 0:
            # One or more equal differences detected -> specialize all of them
            condition = Condition()

            for idx, item in enumerate(self):
                if len(item) > 1:
                    condition[idx] = perception[idx]
        else:
            # Nothing for specialization found
            return None

        return condition
