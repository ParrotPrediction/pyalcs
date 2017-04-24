from alcs.agent import Perception
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

        # Experience
        self.exp = 1

        # When ALP learning was triggered
        self.talp = None

        self.tga = 0

        # Application average
        self.tav = None

    @property
    def fitness(self):
        return self.q * self.r

    def does_anticipate_change(self):
        """
        :return: true if the effect part contains any specified attributes
        """
        return self.effect.number_of_specified_elements > 0

    def does_anticipate_correctly(self, previous_situation: Perception, situation: Perception):
        return self.effect.does_anticipate_correctly()

    def set_alp_timestamp(self, time: int) -> None:
        """
        Sets the ALP time stamp and the application average parameter.
        :param time: current step
        """
        if 1. / self.exp > c.BETA:
            self.tav = (self.tav * self.exp + (time - self.talp)) / (self.exp + 1)
        else:
            self.tav = c.BETA * ((time - self.talp) - self.tav)

        self.talp = time

    def update_reward(self, p: float) -> float:
        self.r += c.BETA * (p - self.r)
        return self.r

    def update_intermediate_reward(self, rho) -> float:
        self.ir += c.BETA * (rho - self.ir)
        return self.ir

    def increase_experience(self) -> int:
        self.exp += 1
        return self.exp
