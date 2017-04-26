from alcs.agent import Perception
from alcs.agent.acs3 import Condition, Action, Effect
from alcs.agent.acs3 import Constants as c


class Classifier(object):

    def __init__(self,
                 condition=None,
                 action=None,
                 effect=None,
                 quality=0.5,
                 reward=0,
                 intermediate_reward=0,
                 numerosity=1,
                 experience=1):

        self.condition = Condition(condition) if condition is not None else Condition()
        self.action = Action(action) if action is not None else None
        self.effect = Effect(effect) if effect is not None else Effect()

        # Quality - measures the accuracy of the anticipations
        self.q = quality

        # The reward prediction - predicts the reward expected after
        # the execution of action A given condition C
        self.r = reward

        # Intermediate reward
        self.ir = intermediate_reward

        # Numerosity
        self.num = numerosity

        # Experience
        self.exp = experience

        # When ALP learning was triggered
        self.talp = None

        self.tga = 0

        # Application average
        self.tav = None

    @classmethod
    def cover_triple(cls,
                     previous_situation: Perception,
                     action: int,
                     situation: Perception,
                     time: int):
        """
        Creates a classifier that anticipates the change correctly

        :param previous_situation:
        :param action:
        :param situation:
        :param time:

        :return: new classifier
        """
        effect = Effect()
        condition = effect.get_and_specialize(previous_situation, situation)

        new_cl = cls(
            condition=condition,
            action=action,
            effect=effect,
            reward=0.5)

        new_cl.pmark = None
        new_cl.tga = time
        new_cl.talp = time
        new_cl.tav = 0

        return new_cl

    @property
    def fitness(self):
        return self.q * self.r

    def does_anticipate_change(self):
        """
        :return: true if the effect part contains any specified attributes
        """
        return self.effect.number_of_specified_elements > 0

    def does_anticipate_correctly(self, previous_situation: Perception, situation: Perception) -> bool:
        return self.effect.does_anticipate_correctly(
            previous_situation, situation)

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

    def expected_case(self, previous_perception: Perception, time: int):
        # TODO: NYI
        pass

    def unexpected_case(self, previous_perception: Perception, perception: Perception, time: int):
        # TODO: NYI
        pass
