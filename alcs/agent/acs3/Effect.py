from alcs.agent.acs2 import Constants as c

from . import Condition


class Effect(object):
    """
    Anticipates the effects that the classifier 'believes'
    to be caused by the specified action.
    """
    def __init__(self):
        self.list = [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH

    def get_specificity(self) -> int:
        return len(self.list)  # Eee. Check it. Seems too simple
