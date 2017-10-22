class TestRandom:
    def __init__(self, values):
        self.values = values
        self.iter = iter(values)

    def __call__(self, *args, **kwargs):
        return next(self.iter)
