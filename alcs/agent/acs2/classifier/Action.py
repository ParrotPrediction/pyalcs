class Action(object):
    """
    Proposes an available action
    """

    def __init__(self, action: int = None):
        self.action = action

    @property
    def action(self):
        return self.action

    @action.setter
    def action(self, value):
        self._action = value
