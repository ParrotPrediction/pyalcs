class Perception(object):

    def __init__(self, perception):
        self.list = perception

    def __getitem__(self, item):
        return self.list[item]

    def __eq__(self, other):
        if other.list == self.list:
            return True

        return False
