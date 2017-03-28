from copy import deepcopy

from . import Constants as c


class Classifier(object):
    """
    Represents a condition-action-effect rule that anticipates the model
    state resulting from the execution of the specified action given the
    specified condition.

    Always specifies a complete resulting state.
    """

    def __init__(self):
        # The Condition Part - specifies the set of situations (perceptions)
        # in which the classifier can be applied
        self.condition = [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH

        # The Action Part - proposes an available action
        self.action = None

        # The Effect Part - anticipates the effects that the classifier
        # 'believes' to be caused by the specified action
        self.effect = [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH

        # The Mark - records the properties in which the classifier did
        # not work correctly before
        self.mark = Classifier.empty_mark()

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
        return 'Classifier{{{}-{}-{} q:{:.2f}, r:{:.2f}, ir:{:.2f}}}'.format(
            ''.join(map(str, self.condition)),
            self.action, ''.join(map(str, self.effect)),
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

    def get_condition_specificity(self) -> float:
        """
        Return information what percentage of condition elements are
        specific.
        :return: number from [0,1]
        """
        specific = sum(1 for e in self.condition if e != c.CLASSIFIER_WILDCARD)
        return specific / c.CLASSIFIER_LENGTH

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

    def set_mark(self, previous_perception: list, perception: list):
        """
        Function sets compares previous and current perception with
        classifier anticipations and sets mark accordingly.

        :param previous_perception: perception in previous trial
        :param perception: obtained perception
        """
        for i, (condition, anticipation, perception, previous_perception)\
            in enumerate(zip(self.condition, self.effect, previous_perception,
                             perception)):

            # print("PP: [{}] => P: [{}] | E: [{}] ".format(previous_perception,
            #                                               perception,
            #                                               anticipation))

            if previous_perception == perception:
                # classifier predicted correctly
                if anticipation == c.CLASSIFIER_WILDCARD:
                    pass
                else:
                    # Effect part required a change
                    self.mark[i].add(previous_perception)
            else:
                # classifier did not predict correctly
                self.mark[i].add(previous_perception)

    @staticmethod
    def is_marked(mark: list) -> bool:
        """
        Returns information whether classifier is marked.

        :param mark: list of sets containing mark
        :return: True if is marked, false otherwise
        """
        for m in mark:
            if len(m) > 0:
                return True

        return False

    @staticmethod
    def empty_mark():
        """
        Return a collection of empty marks
        """
        return [set() for _ in range(c.CLASSIFIER_LENGTH)]
