from agent.acs2 import Constants as const


class Classifier(object):

    def __init__(self):
        self.condition = [const.CLASSIFIER_WILDCARD] * const.CLASSIFIER_LENGTH
        self.action = None
        self.effect = [const.CLASSIFIER_WILDCARD] * const.CLASSIFIER_LENGTH
        self.mark = None  # All cases were effect was not good
        self.q = 0.5  # quality
        self.r = 0  # reward

        self.ir = 0  # Immediate reward
        self.t = None  # Has been created at this time
        self.tga = 0  # Last time that self was part of action set within GA
        self.alp = 0  # Last time that self underwent ALP process

        self.aav = 0  # Application average
        self.exp = 0  # Experience
        self.num = 1  # Still Micro/macro stuff

    def __copy__(self):
        raise NotImplementedError('Not yet implemented')

    def __eq__(self, other):
        """
        Equality check. The other classifier is the same when it has the same condition and action part

        :param other: the other classifier
        :return: true if classifier is the same, false otherwise
        """
        if isinstance(other, self.__class__):
            if other.condition == self.condition:
                if other.action == self.action:
                    return True

        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_subsumer(self, cl, theta_exp=None, theta_r=None):
        """
        Subsumption - elimination of accurate, over-specialized classifiers
        """
        if not isinstance(cl, self.__class__):
            raise TypeError('Illegal type of classifier passed')

        if theta_exp is None:
            theta_exp = const.THETA_EXP

        if theta_r is None:
            theta_r = const.THETA_R

        cp = 0
        cpt = 0

        if self.exp > theta_exp and self.q > theta_r and self.mark is None:
            for i in range(const.CLASSIFIER_LENGTH):
                if self.condition[i] == const.CLASSIFIER_WILDCARD:
                    cp += 1

                if cl.condition[i] == const.CLASSIFIER_WILDCARD:
                    cpt += 1

            if cp <= cpt:
                if self.effect == cl.effect:
                    return True

        return False

    def is_more_general(self, cl):
        if not isinstance(cl, self.__class__):
            raise TypeError('Illegal type of classifier passed')

        ret = False

        for i in range(const.CLASSIFIER_LENGTH):
            if self.condition[i] != const.CLASSIFIER_WILDCARD and self.condition[i] != cl.condition[i]:
                return False
            elif self.condition[i] != cl.condition[i]:
                ret = True

        return ret
