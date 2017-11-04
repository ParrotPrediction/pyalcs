from random import random, sample

from alcs import Perception
from alcs.acs2 import Condition, Effect, PMark, ACS2Configuration


class Classifier(object):
    def __init__(self,
                 condition=None,
                 action=None,
                 effect=None,
                 quality=0.5,
                 reward=0.5,
                 intermediate_reward=0,
                 numerosity=1,
                 experience=1,
                 talp=None,
                 tga=0,
                 tav=0,
                 cfg=None):

        self.cfg = cfg or ACS2Configuration.default()
        self.condition = Condition(condition, cfg=self.cfg) \
            if condition is not None else Condition(cfg=self.cfg)
        self.action = action if action is not None else None
        self.effect = Effect(effect, cfg=self.cfg) \
            if effect is not None else Effect(cfg=self.cfg)
        self.mark = PMark(cfg=self.cfg)

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
        self.talp = talp

        self.tga = tga

        # Application average
        self.tav = tav

        # I don't know yet what it is
        self.ee = 0

    def q3num(self):
        return pow(self.q, 3) * self.num

    def __repr__(self):
        return "{}-{}-{} @ {}".format(self.condition,
                                      self.action,
                                      self.effect,
                                      hex(id(self)))

    @classmethod
    def copy_from(cls, old_cls, time):
        """
        Copies old classifier with given time (tga, talp).
        Old tav gets replaced with new value.
        New classifier also has no mark.

        :param old_cls: classifier to copy from
        :param time: time of creation
        :return: new classifier
        """
        new_cls = cls(
            condition=Condition(old_cls.condition),
            action=old_cls.action,
            effect=old_cls.effect,
            quality=old_cls.q,
            reward=old_cls.r,
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
        Creates a classifier that anticipates a change correctly

        :param previous_situation:
        :param action:
        :param situation:
        :param time:

        :return: new classifier
        """
        new_cl = cls(action=action)  # TODO: p5 exp=0, r=0 (paper)
        new_cl.tga = time
        new_cl.talp = time

        new_cl.specialize(previous_situation, situation)

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
            if cpi != self.cfg.classifier_wildcard and \
                    epi == self.cfg.classifier_wildcard:
                spec += 1

        return spec

    @property
    def specificity(self):
        return self.condition.specificity / len(self.condition)

    def does_anticipate_change(self) -> bool:
        """
        :return: true if the effect part contains any specified attributes
        """
        return self.effect.number_of_specified_elements > 0

    def is_reliable(self):
        return self.q > self.cfg.theta_r

    def is_inadequate(self):
        return self.q < self.cfg.theta_i

    def update_reward(self, p: float) -> float:
        self.r += self.cfg.beta * (p - self.r)
        return self.r

    def update_intermediate_reward(self, rho) -> float:
        self.ir += self.cfg.beta * (rho - self.ir)
        return self.ir

    def increase_experience(self) -> int:
        self.exp += 1
        return self.exp

    def increase_quality(self) -> float:
        self.q += self.cfg.beta * (1 - self.q)
        return self.q

    def decrease_quality(self) -> float:
        self.q -= self.cfg.beta * self.q
        return self.q

    def specialize(self,
                   previous_situation: Perception,
                   situation: Perception):
        """
        Specializes the effect part where necessary to correctly anticipate
        the changes from p0 to p1 and returns a condition which specifies
        the attributes which must be specified in the condition part.
        The specific attributes in the returned conditions are set to
        the necessary values.

        :param previous_situation:
        :param situation:
        """
        for idx, item in enumerate(situation):
            if previous_situation[idx] != situation[idx]:
                self.effect[idx] = situation[idx]
                self.condition[idx] = previous_situation[idx]

    def predicts_successfully(self,
                              p0: Perception,
                              action: int,
                              p1: Perception) -> bool:
        """
        Check if classifier matches previous situation `p0`,
        has action `action` and predicts the effect `p1`

        :param p0: previous situation
        :param action:
        :param p1: anticipated situation after execution action
        :return: True if classifier makes successful predictions,
        False otherwise
        """
        if self.condition.does_match(p0):
            if self.action == action:
                if self.does_anticipate_correctly(p0, p1):
                    return True

        return False

    def does_anticipate_correctly(self,
                                  previous_situation: Perception,
                                  situation: Perception) -> bool:
        return self.effect.does_anticipate_correctly(
            previous_situation, situation)

    def set_mark(self, perception: Perception) -> None:
        """
        Marks classifier with given perception taking into consideration its
        condition.

        Specializes the mark in all attributes which are not specified
        in the conditions, yet

        :param perception: current situation
        """
        if self.mark.set_mark_using_condition(self.condition, perception):
            self.ee = 0

    def set_alp_timestamp(self, time: int) -> None:
        """
        Sets the ALP time stamp and the application average parameter.

        :param time: current step
        """
        # TODO p5: write test
        if 1. / self.exp > self.cfg.beta:
            self.tav = (self.tav * self.exp + (time - self.talp)) / (
                self.exp + 1)
        else:
            self.tav += self.cfg.beta * ((time - self.talp) - self.tav)

        self.talp = time

    def expected_case(self,
                      perception: Perception,
                      time: int):
        """
        Controls the expected case of a classifier. If the classifier
        is to specific it tries to add some randomness to it by
        generalizing some attributes.

        :param perception:
        :param time:
        :return: new classifier or None
        """
        diff = self.mark.get_differences(perception)

        if diff is None:
            self.increase_quality()
            return None

        no_spec = self.specified_unchanging_attributes
        no_spec_new = diff.specificity
        child = self.copy_from(self, time)

        if no_spec >= self.cfg.u_max:
            # TODO: p4: implement later
            # Code below won't get executed anyway because c.U_MAX is high
            pass
        else:
            # TODO: p4: implement later
            pass

        child.condition.specialize(new_condition=diff)

        if child.q < 0.5:
            child.q = 0.5

        return child

    def unexpected_case(self,
                        previous_perception: Perception,
                        perception: Perception,
                        time: int):
        """
        Controls the unexpected case of the classifier.

        :param previous_perception:
        :param perception:
        :param time:
        :return: specialized classifier if generation was possible,
        None otherwise
        """
        self.decrease_quality()
        self.set_mark(previous_perception)

        # Return if the effect is not specializable
        if not self.effect.is_specializable(previous_perception, perception):
            return None

        child = self.copy_from(self, time)

        # TODO: p5 maybe also take into consideration cl.E = # (paper)
        child.specialize(previous_perception, perception)

        # TODO: p5 cl.q? (paper)
        if child.q < 0.5:
            child.q = 0.5

        return child

    def mutate(self, randomfunc=random):
        """
        Executes the generalizing mutation in the classifier.
        """
        for idx, cond in enumerate(self.condition):
            if cond != self.cfg.classifier_wildcard and \
                    randomfunc() < self.cfg.mu:
                self.condition.generalize(idx)

    def is_similar(self, other) -> bool:
        """
        Check if classifier is equals to `other` classifier in condition,
        action and effect part.

        :param other: other classifier
        :return: True if equals, False otherwise
        """
        if self.condition == other.condition and \
                self.action == other.action and \
                self.effect == other.effect:
            return True
        return False

    def is_equally_general(self, other) -> bool:
        return self.condition.specificity == other.condition.specificity

    def is_more_general(self, other) -> bool:
        """
        Checks if the classifier is formally more general than `other`.

        :param other: other classifier to compare
        :return: True if `other` classifier is more general
        """
        if self.condition.specificity < other.condition.specificity:
            return True

        return False

    def does_subsume(self, other) -> bool:
        """
        Returns if a classifier subsumes `other` classifier

        :param other: other classifiers
        :return: True if `other` classifier is subsumed, False otherwise
        """
        if self._is_subsumer() and \
            self.is_more_general(other) and \
            self.condition.does_match(other.condition) and \
                self.effect == other.effect:
            return True

        return False

    def _is_subsumer(self) -> bool:
        """
        Controls if the classifier satisfies the subsumer criteria.

        :return: True is classifier can be considered as subsumer,
        False otherwise
        """
        if self.exp > self.cfg.theta_exp:
            if self.is_reliable():
                if self.mark.is_empty():
                    return True

        return False

    def is_unmarked(self):
        return not self.is_marked()

    def is_marked(self):
        """
        Returns if classifier is marked
        """
        if self.mark.is_empty():
            return False

        return True

    def crossover(self, cl2, samplefunc=sample):
        """
        Executes crossover with other classifier
        :param samplefunc:
        :param cl2: other classifier
        """
        self.condition.two_point_crossover(
            cl2.condition, samplefunc=samplefunc)

        q = float(sum([self.q, cl2.q]) / 2)
        r = float(sum([self.r, cl2.r]) / 2)

        cl2.q = q
        cl2.r = r

    def __eq__(self, other):
        return self.condition == other.condition \
            and self.effect == other.effect \
            and self.action == other.action \
            and self.num == other.num \
            and self.r == other.r \
            and self.q == other.q \
            and self.talp == other.talp \
            and self.tav == other.tav \
            and self.tga == other.tga \
            and self.exp == other.exp \
            and self.ee == other.ee \
            and self.mark == other.mark \
            and self.ir == other.ir
