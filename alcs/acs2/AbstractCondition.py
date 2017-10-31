from alcs import Perception
from alcs.acs2 import default_configuration


class AbstractCondition(list):
    def __init__(self, seq=(), cfg=None):
        self.cfg = cfg or default_configuration
        if not seq:
            list.__init__(
                self,
                [self.cfg.classifier_wildcard] * self.cfg.classifier_length)
        else:
            list.__init__(self, seq)
            if len(self) != self.cfg.classifier_length:
                raise ValueError('Illegal length of effect string')

    def __setitem__(self, idx, value):
        if not isinstance(value, str):
            raise TypeError('Effect should be composed of string objects')

        super(AbstractCondition, self).__setitem__(idx, value)

    def __repr__(self):
        return ''.join(map(str, self))

    def is_specializable(self, p0: Perception, p1: Perception) -> bool:
        """
        Determines if the effect part can be modified to anticipate
        changes from `p0` to `p1` correctly by only specializing attributes.

        :param p0: previous perception
        :param p1: current perception
        :return: True if specializable, false otherwise
        """
        for p0i, p1i, ei in zip(p0, p1, self):
            if ei != self.cfg.classifier_wildcard:
                if ei != p1i or p0i == p1i:
                    return False

        return True
