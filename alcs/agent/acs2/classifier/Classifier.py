from copy import deepcopy

from alcs.agent.acs2 import Constants as c
from alcs.environment.maze import MazeAction

from . import Mark, Action, Condition, Effect
from ..Perception import Perception


class Classifier(object):
    """
    Represents a condition-action-effect rule that anticipates the model
    state resulting from the execution of the specified action given the
    specified condition.

    Always specifies a complete resulting state.
    """

    def __init__(self):
        self.condition = Condition()
        self.action = Action()
        self.effect = Effect()
        self.mark = Mark()

        # Quality - measures the accuracy of the anticipations
        self.q = 0.5

        # The reward prediction - predicts the reward expected after
        # the execution of action A given condition C
        self.r = 0

        # The immediate reward prediction - predicts the reinforcement
        # directly encountered after the execution of action A
        self.ir = 0

        # In which generation the classifier was created
        self.t = None

        # The GA timestamp - records the last time the classifier was part
        # of an action set in which GA was applied
        self.t_ga = 0

        # The ALP timestamp - records the time the classifier underwent
        # the last ALP update
        self.t_alp = 0

        # The 'application average' - estimates the ALP update frequency
        self.aav = 0

        # The 'experience counter' - counts the number of times the classifier
        # underwent the ALP
        self.exp = 0

        # The numerosity - specifies the number of actual (micro-) classifier
        # this macroclassifier represents
        self.num = 1

    @staticmethod
    def copy_from(old_classifier):
        return deepcopy(old_classifier)

    def __repr__(self):
        return 'Classifier{{{} {} {} q:{:.2f}, r:{:.2f}, ir:{:.2f}}}'.format(
            ''.join(map(str, self.condition)),
            MazeAction().find_symbol(self.action),
            ''.join(map(str, self.effect)),
            self.q,
            self.r,
            self.ir
        )

    def __eq__(self, other):
        """
        Equality check. The other classifier is the same when
        it has the same condition, action and effect part

        :param other: the other classifier
        :return: true if classifier is the same, false otherwise
        """
        if isinstance(other, self.__class__):
            if (other.condition == self.condition and
                    other.action == self.action and
                    other.effect == self.effect):
                return True

        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.condition, self.action, self.effect))


    #####
    def expected_case(self, percept: Perception, time: int):
        pass

    def unexpected_case(self, p0: Perception, p1: Perception, time: int):
        pass

    def is_similar(self, cls):
        pass

    def does_match(self, percept: Perception) -> bool:
        pass

    def does_match_backwards(self, percept: Perception) -> bool:
        pass

    def does_link(self, cls) -> bool:
        pass

    def has_action(self, action: int) -> bool:
        pass

    def does_anticipate_correct(self, p0: Perception, p1: Perception) -> bool:
        pass

    def does_subsume(self, cls) -> bool:
        pass

    def mutate(self):
        pass

    def crossover(self, cls):
        pass



    #####

    def fitness(self):
        return self.q * self.r

    def is_reliable(self) -> bool:
        """
        Returns information whether a classifier
        can be considered as reliable
        """
        return self.q > c.THETA_R

    def is_inadequate(self) -> bool:
        """
        Returns information whether a classifier
        can be considered as inadequate.
        """
        return self.q < c.THETA_I

    def get_micro_classifiers(self) -> []:
        """
        Returns an array of self-contained micro-classifiers (according
        to numerosity).

        :return: array of classifiers
        """
        micros = []

        for micro_cl in range(0, self.num):
            micros.append(self)

        return micros

    def can_subsume(self, cl_tos, theta_exp=None, theta_r=None):
        """
        Subsume operation - capture another, similar but more
        general classifier.

        In order to subsume another classifier, the subsumer needs to be
        experienced, reliable and not marked. Moreover the subsumer condition
        part needs to be syntactically more general and the effect part
        needs to be identical

        :param cl_tos: classifier to subsume
        :param theta_exp: threshold of required classifier experience
        to subsume another classifier
        :param theta_r: threshold of required classifier quality to
        subsume another classifier
        :return: true if classifier cl is subsumed, false otherwise
        """
        if not isinstance(cl_tos, self.__class__):
            raise TypeError('Illegal type of classifier passed')

        if theta_exp is None:
            theta_exp = c.THETA_EXP

        if theta_r is None:
            theta_r = c.THETA_R

        cp = 0  # number of subsumer wildcards in condition part
        cpt = 0  # number of wildcards in condition part in other classifier

        if (self.exp > theta_exp and
                self.q > theta_r and
                not __class__.is_marked(self.mark)):

            for i in range(c.CLASSIFIER_LENGTH):
                if self.condition[i] == c.CLASSIFIER_WILDCARD:
                    cp += 1

                if cl_tos.condition[i] == c.CLASSIFIER_WILDCARD:
                    cpt += 1

            if cp <= cpt:
                if self.effect == cl_tos.effect:
                    return True

        return False

    def is_more_general(self, cl):
        """
        Checks if classifier is more general than classifier passed in
        an argument. It's made sure that classifier is indeed *more* general,
        as well as that the more specific classifier is completely included
        in the more general one (do not specify overlapping regions).

        :param cl: classifier to compare
        :return: true if a base classifier is more general, false otherwise
        """
        if not isinstance(cl, self.__class__):
            raise TypeError('Illegal type of classifier passed')

        base_more_general = False

        for i in range(c.CLASSIFIER_LENGTH):
            if (self.condition[i] != c.CLASSIFIER_WILDCARD and
                    self.condition[i] != cl.condition[i]):

                return False
            elif self.condition[i] != cl.condition[i]:
                base_more_general = True

        return base_more_general
