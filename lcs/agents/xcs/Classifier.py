
from typing import Union, Optional
from lcs import Perception

from lcs.agents.xcs import Configuration, Condition


class Classifier:
    def __init__(self,
                 cfg: Optional[Configuration] = None,
                 condition: Union[Condition, str, None] = None,
                 action: Optional[int] = None,
                 time_stamp: int = None) -> None:
        if cfg is None:
            raise TypeError("Configuration should be passed to Classifier")
        if type(condition) != Condition:
            condition = str(condition)
        self.cfg = cfg
        self.condition = condition
        self.action = action
        self.time_stamp = time_stamp
        self.experience = 0
        self.action_set_size = 1
        self.numerosity = 1
        self.prediction, self.error, self.fitness \
            = cfg.initial_classifier_values()

    def does_match(self, situation):
        if len(situation) != len(self):
            return False
        return self.condition.subsumes(situation)

    def does_subsume(self, other):
        if self.action == other.action and \
           self.could_subsume() and \
           self.is_more_general(other):
                    return True
        return False

    def could_subsume(self):
        if self.experience > self.cfg.theta_sub and self.error < self.cfg.epsilon_i:
                return True
        return False

    def is_more_general(self, other):
        if self.wildcard_number() <= other.wildcard_number():
            return False
        return self.condition.is_more_general(other.condition)

    def wildcard_number(self):
        return self.condition.wildcard_number()

    def __eq__(self, other):
        if type(other) != Classifier:
            raise TypeError("Classifier can only = other classifiers")
        if self.does_match(other.condition) and self.action == other.action:
            return True
        return False

    def __len__(self):
        return len(self.condition)
