from alcs.agent.acs3 import Constants as c
from alcs.agent import Perception
from alcs.agent.acs3 import Condition


class PMark(list):
    def __init__(self):
        list.__init__(self, [set() for _ in range(c.CLASSIFIER_LENGTH)])

    def __setitem__(self, idx, value):
        if not isinstance(value, str):
            raise TypeError('Mark should be composed of string objects')

        self[idx].add(value)

    def set_mark(self, perception: Perception):
        """
        Directly further specializes all specified attributes in the mark

        :param perception:
        :return: True if something was specialized
        """
        changed = False

        for idx, item in enumerate(perception):
            if item not in self[idx]:
                changed = True
                self[idx] = item

        return changed

    def is_empty(self) -> bool:
        """
        Check if there is any mark

        :return: True if is marked, False otherwise
        """
        return not any([True for m in self if len(m) > 0])

    def get_differences(self, perception: Perception) -> Condition:
        """
        Determines the strongest differences in between the mark
        and perception.

        :param: perception
        :return: condition that specifies all the differences.
        """
        condition = None
        nr1 = 0
        nr2 = 0

        # Count difference types
        for idx, item in enumerate(self):
            if perception[idx] not in item:
                nr1 += 1
            elif len(item) > 1:
                nr2 += 1

        if nr1 > 0:
            # One or more absolute differences detected -> specialize one
            # randomly chosen
            pass
        elif nr2 > 0:
            # One or more equal differences detected -> specialize all of them
            pass
        else:
            # Nothing for specialization found
            pass

        # TODO: NYI
        return condition
