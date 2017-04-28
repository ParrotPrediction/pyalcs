from alcs.agent import Perception
from alcs.agent.acs3 import Condition, Action, Effect, PMark
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
        self.mark = PMark()

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

        # I don't know yet what it is
        self.ee = 0

    @classmethod
    def copy_from(cls, old_cls, time):
        """
        Copies old classifier with given time.
        New classifier has no mark.

        :param old_cls: classifier to copy from
        :param time:
        :return: new classifier
        """
        new_cls = cls(
            condition=old_cls.condition,
            action=old_cls.action.action,
            effect=old_cls.effect,
            quality=old_cls.q,
            reward=old_cls.ir,
            intermediate_reward=old_cls.ir)

        new_cls.tga = time
        new_cls.talp = time
        new_cls.tav = old_cls.tav

        return new_cls

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

    @property
    def specified_unchanging_attributes(self):
        """
        Determines the number of specified unchanging attributes in
        the classifier. An unchanging attribute is one that is anticipated
        not to change in the effect part.

        :return: number of specified unchanging attributes
        """
        spec = 0

        for cpi, epi in zip(self.condition, self.effect):
            if cpi != c.CLASSIFIER_WILDCARD and epi == c.CLASSIFIER_WILDCARD:
                spec += 1

        return spec

    def does_anticipate_change(self):
        """
        :return: true if the effect part contains any specified attributes
        """
        return self.effect.number_of_specified_elements > 0

    def does_anticipate_correctly(self, previous_situation: Perception, situation: Perception) -> bool:
        return self.effect.does_anticipate_correctly(
            previous_situation, situation)

    def set_mark(self, perception: Perception) -> None:
        """
        Marks classifier with given perception taking into consideration its
        condition.

        :param perception:
        """
        if self.mark.set_mark(perception):
            self.ee = 0

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

    def increase_quality(self) -> float:
        self.q += c.BETA * (1 - self.q)
        return self.q

    def decrease_quality(self) -> float:
        self.q -= c.BETA * self.q
        return self.q

    def expected_case(self, previous_perception: Perception, time: int):
        """
        Controls the expected case of a classifier. If the classifier
        is to specific it tries to add some randomness to it by
        generalizing some attributes.

        :param previous_perception:
        :param time:
        :return: new classifier or None
        """
        diff = self.mark.get_differences(previous_perception)

        if diff is None:
            self.increase_quality()
            return

        cl = self.copy_from(self, time)
        no_spec = self.specified_unchanging_attributes
        no_spec_new = diff.number_of_specified_elements

        # TODO: implement later
        # Code below won't get executed anyway because c.U_MAX is high
        if no_spec >= c.U_MAX:
            pass
        else:
            pass

        if cl.q < 0.5:
            cl.q = 0.5

        return cl

    def unexpected_case(self,
                        previous_perception: Perception,
                        perception: Perception,
                        time: int):
        """
        Controls the unexpected case of the classifier.

        :param previous_perception:
        :param perception:
        :param time:
        :return: specialized classifier if generation was possible, None otherwise
        """
        self.decrease_quality()
        self.set_mark(previous_perception)

        if self.effect.is_specializable(previous_perception, perception):
            cl = self.copy_from(self, time)

            diff = cl.effect.get_and_specialize(previous_perception, perception)
            cl.condition.specialize(new_condition=diff)

            if cl.q < 0.5:
                cl.q = 0.5

            return cl

        return None
