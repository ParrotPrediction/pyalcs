from alcs.agent.Perception import Perception
from alcs.agent.acs2 import Constants as c


class Mark(object):
    """
    Records the properties in which the classifier did not work correctly
    before.
    """

    def __init__(self):
        self.list = [set() for _ in range(c.CLASSIFIER_LENGTH)]

    def is_empty(self) -> bool:
        """
        Returns information whether classifier is marked.

        :return: True if is marked, false otherwise
        """
        for m in self.list:
            if len(m) > 0:
                return False

        return True

    def set_mark(self, perception: Perception):
        """
        Directly further specializes all specified attributes in the mark

        :param perception:
        """
        for idx, m in enumerate(self.list):
            m.add(perception[idx])

    def set_mark(self, condition, perception: Perception):
        """
        Specializes the mark in all attributes which are not specified
        in the conditions, yet.

        :param condition:
        :param perception:
        :return: if the mark was actually specialized (enhanced)
        """
        if len(self.list) != 0:
            self.set_mark(perception)
