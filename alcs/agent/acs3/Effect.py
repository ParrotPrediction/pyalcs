from alcs.agent.acs2 import Constants as c


class Effect(list):
    """
    Anticipates the effects that the classifier 'believes'
    to be caused by the specified action.
    """
    def __init__(self, *args):
        if len(args) > 0:
            list.__init__(self, *args)
        else:
            list.__init__(self, [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH)

    def __setitem__(self, idx, value):
        if not isinstance(value, str):
            raise TypeError('Effect should be composed of string objects')

        super(Effect, self).__setitem__(idx, value)

    def __repr__(self):
        return ''.join(map(str, self))

    @property
    def number_of_specified_elements(self) -> int:
        """
        :return: number of specific components
        """
        return sum(1 for comp in self if comp != c.CLASSIFIER_WILDCARD)
