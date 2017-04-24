# TODO: action setter should be limited to int and within environment range
from functools import total_ordering


@total_ordering
class Action(object):

    def __init__(self, action: int):
        self._action = action

    def __repr__(self):
        return "{}".format(self._action)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if other.action == self.action:
                return True

        return False

    def __lt__(self, other):
        return self.action < other.action

    @property
    def action(self):
        return self._action
