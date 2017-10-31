class TestRandom:
    def __init__(self, values):
        self.values = values
        self.iter = iter(values)

    def __call__(self, *args, **kwargs):
        return next(self.iter)


class TestSample:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], TestRandom):
            self.random_stream = args[0]
        else:
            self.random_stream = TestRandom(args[0])

    def __call__(self, *args, **kwargs):
        assert args is not None and len(args) == 2
        lst = list(args[0])
        sample_size = int(args[1])
        population_size = len(lst)
        result = []
        already_used = set()
        for i in range(sample_size):
            rnd = self.random_stream()
            elem_id = rnd % population_size
            while elem_id in already_used:
                rnd = self.random_stream()
                elem_id = rnd % population_size
            result.append(lst[elem_id])
            already_used.add(elem_id)
        return result
