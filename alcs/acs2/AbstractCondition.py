from alcs.acs2 import ACS2Configuration


class AbstractCondition(list):
    def __init__(self, seq=(), cfg=None):
        self.cfg = cfg or ACS2Configuration.default()
        if not seq:
            list.__init__(
                self,
                [self.cfg.classifier_wildcard] * self.cfg.classifier_length)
        else:
            list.__init__(self, seq)
            if len(self) != self.cfg.classifier_length:
                raise ValueError('Illegal length of perception string')
            if isinstance(seq, AbstractCondition) and cfg is None:
                self.cfg = seq.cfg

    def __setitem__(self, idx, value):
        if not isinstance(value, str):
            raise TypeError('Perception should be composed of string objects')

        super(AbstractCondition, self).__setitem__(idx, value)

    def __repr__(self):
        return ''.join(map(str, self))
