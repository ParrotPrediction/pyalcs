from alcs.agent.acs3.Condition import Condition
from alcs.agent.acs3.Effect import Effect


class Classifier(object):

    def __init__(self):
        self.condition = Condition()
        self.effect = Effect()

        # Quality - measures the accuracy of the anticipations
        self.q = 0.5

        # The reward prediction - predicts the reward expected after
        # the execution of action A given condition C
        self.r = 0

    @property
    def fitness(self):
        return self.q * self.r

    def does_anticipate_change(self):
        return self.effect.get_specificity()

