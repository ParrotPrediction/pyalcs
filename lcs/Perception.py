class Perception(list):
    def __init__(self, *args):
        list.__init__(self, *args)

    def __repr__(self):
        return ''.join(map(str, self))
