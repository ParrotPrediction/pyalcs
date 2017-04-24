from alcs.agent.acs3 import Condition, Action, Effect
from alcs.agent.acs3 import Constants as c


class Classifier(object):

    def __init__(self, action=None):
        self.condition = Condition()
        self.action = Action(action)
        self.effect = Effect()

        # Quality - measures the accuracy of the anticipations
        self.q = 0.5

        # The reward prediction - predicts the reward expected after
        # the execution of action A given condition C
        self.r = 0

        # Intermediate reward
        self.ir = 0

        # Numerosity
        self.num = 1

        # When ALP learning was triggered
        self.talp = None

    @property
    def fitness(self):
        return self.q * self.r

    def does_anticipate_change(self):
        """
        :return: true if the effect part contains any specified attributes
        """
        return self.effect.number_of_specified_elements > 0

    def update_reward(self, p: float) -> float:
        self.r += c.BETA * (p - self.r)
        return self.r

    def update_intermediate_reward(self, rho) -> float:
        self.ir += c.BETA * (rho - self.ir)
        return self.ir
