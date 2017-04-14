from alcs.agent.acs2 import Constants as c

from . import Condition
from .. import Perception


class Effect(object):
    """
    Anticipates the effects that the classifier 'believes'
    to be caused by the specified action.
    """
    def __init__(self):
        self.list = [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH

    def __eq__(self, other):
        pass

    def get_best_anticipation(self, perception: Perception) -> Perception:
        pass

    def get_best_anticipation(self, condition: Condition) -> Condition:
        pass

    def does_anticipate_correctly(self, p0: Perception, p1: Perception) -> bool:
        pass

    def is_enhanced(self) -> bool:
        pass

    def update_enhanced_effect_probs(self, percept: Perception, update_rate: float):
        pass

    def does_match(self, perception: Perception, condPerception: Perception) -> bool:
        pass

    def does_specify_only_changes_backwards(self, backAnt: Perception, situation: Perception) -> bool:
        pass

    def is_specializable(self, p0: Perception, p1: Perception) -> bool:
        pass

    def get_specificity(self) -> int:
        pass
