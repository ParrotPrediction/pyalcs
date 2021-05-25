import logging
from typing import Union, Optional

from lcs.agents.xcs import Configuration, Condition

logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self,
                 cfg: Optional[Configuration] = None,
                 condition: Union[Condition, str, None] = None,
                 action: Optional[int] = None,
                 time_stamp: int = None) -> None:

        if cfg is None:
            raise TypeError("Configuration should be passed to Classifier")

        if type(condition) != Condition:
            condition = Condition(condition)

        self.cfg = cfg                  # cfg
        self.condition = condition      # current situation
        self.action = action            # A - int action
        self.time_stamp = time_stamp    # time - int
        self.experience = 0             # exp
        self.action_set_size = 1        # as
        self.numerosity = 1             # num
        self.prediction = cfg.initial_prediction
        self.error = cfg.initial_error
        self.fitness = cfg.initial_fitness
        # p, Epsilon, f

    def does_match(self, situation):
        if len(situation) != len(self):
            return False
        return self.condition.subsumes(situation)

    def does_subsume(self, other):
        if self.action == other.action and \
           self.could_subsume and \
           self.is_more_general(other):
                    return True
        return False

    @property
    def could_subsume(self):
        return self.experience > self.cfg.subsumption_threshold and self.error < self.cfg.initial_error

    def is_more_general(self, other):
        if self.wildcard_number <= other.wildcard_number:
            return False
        return self.condition.is_more_general(other.condition)

    @property
    def wildcard_number(self):
        return self.condition.wildcard_number

    def __len__(self):
        return len(self.condition)

    def __str__(self):
        return f"Cond:{self.condition} - Act:{self.action} - Num:{self.numerosity} " + \
            f"[fit: {self.fitness:.3f}, exp: {self.experience:3.2f}, pred: {self.prediction:2.3f}, Error:{self.error}]"

    def __eq__(self, o):
        return o.condition == self.condition and o.action == self.action

    def __hash__(self):
        return hash((str(self.condition), self.action))
