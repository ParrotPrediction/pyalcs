from agent.acs2 import Constants as const


class Classifier(object):
    def __init__(self):
        self.condition = [const.CLASSIFIER_WILDCARD] * const.CLASSIFIER_LENGTH
        self.action = None
        self.effect = [const.CLASSIFIER_WILDCARD] * const.CLASSIFIER_LENGTH
        self.q = 0.5  # quality
        self.r = 0  # reward

        self.mark = None  # All cases were effect was not good
        self.ir = 0  # Immediate reward
        self.t = None  # Has been created at this time
        self.tga = 0  # Last time that self was part of action set within GA
        self.alp = 0  # Last time that self underwent

        self.aav = 0  # Application average
        self.exp = 0  # Experience
        self.num = 1  # Still Micro/macro stuff
        self.gold = False  # TODO: is it useful?

        if self.condition == [const.CLASSIFIER_WILDCARD] * const.CLASSIFIER_LENGTH:
            if self.effect == [const.CLASSIFIER_WILDCARD] * const.CLASSIFIER_LENGTH:
                if self.t == 0:
                    self.gold = True
        else:
            self.gold = False

    def __copy__(self):
        raise NotImplementedError('Not yet implemented')

    def is_subsumer(self, cl, theta_exp=None, theta_r=None):
        if not isinstance(cl, Classifier):
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
        if not isinstance(cl, Classifier):
            raise TypeError('Illegal type of classifier passed')

        ret = False

        for i in range(const.CLASSIFIER_LENGTH):
            if self.condition[i] != const.CLASSIFIER_WILDCARD and self.condition[i] != cl.condition[i]:
                return False
            elif self.condition[i] != cl.condition[i]:
                ret = True

        return ret

    def equals(self, cl):
        if not isinstance(cl, Classifier):
            raise TypeError('Illegal type of classifier passed')

        if cl.condition == self.condition:
            if cl.action == self.action:
                return True

        return False
