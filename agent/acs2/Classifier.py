from agent.acs2 import Constants as c


class Classifier(object):

    def __init__(self):
        self.condition = [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH
        self.action = None
        self.effect = [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH
        self.mark = None  # All cases were effect was not good
        self.q = 0.5  # quality
        self.r = 0  # reward

        self.ir = 0  # Immediate reward
        self.t = None  # Has been created at this time
        self.tga = 0  # Last time that self was part of action set within GA
        self.alp = 0  # Last time that self underwent ALP process

        self.aav = 0  # Application average
        self.exp = 0  # Experience
        self.num = 1  # Numerosity (how many classifiers were subsumed)

    def __repr__(self):
        return 'Classifier{{{}-{}-{} q:{}, r:{}}}'.format(
            ''.join(map(str, self.condition)),
            self.action, ''.join(map(str, self.effect)),
            self.q,
            self.r)

    def __copy__(self):
        raise NotImplementedError('Not yet implemented')

    def __eq__(self, other):
        """
        Equality check. The other classifier is the same when
        it has the same condition and action part

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

    def fitness(self):
        return self.q * self.r

    def is_subsumer(self, cl, theta_exp=None, theta_r=None):
        """
        Subsume operation - capture another, similar but more
        general classifier.

        In order to subsume another classifier, the subsumer needs to be
        experienced, reliable and not marked. Moreover the subsumer condition
        part needs to be syntactically more general and the effect part
        needs to be identical

        :param cl: classifier to subsume
        :param theta_exp: threshold of required classifier experience
        to subsume another classifier
        :param theta_r: threshold of required classifier quality to
        subsume another classifier
        :return: true if classifier cl is subsumed, false otherwise
        """
        if not isinstance(cl, self.__class__):
            raise TypeError('Illegal type of classifier passed')

        if theta_exp is None:
            theta_exp = c.THETA_EXP

        if theta_r is None:
            theta_r = c.THETA_R

        cp = 0  # number of subsumer wildcards in condition part
        cpt = 0  # number of wildcards in condition part in other classifier

        if self.exp > theta_exp and self.q > theta_r and self.mark is None:
            for i in range(c.CLASSIFIER_LENGTH):
                if self.condition[i] == c.CLASSIFIER_WILDCARD:
                    cp += 1

                if cl.condition[i] == c.CLASSIFIER_WILDCARD:
                    cpt += 1

            if cp <= cpt:
                if self.effect == cl.effect:
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
