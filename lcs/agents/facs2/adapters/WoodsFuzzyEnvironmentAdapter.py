from .FuzzyEnvironmentAdapter import FuzzyEnvironmentAdapter


class WoodsFuzzyEnvironmentAdapter(FuzzyEnvironmentAdapter):
    _path = 0
    _wall = 1
    _reward = 9
    condition_length = 12

    def __init__(self, env):
        super().__init__(env)
        self._functions = [
            self._generate_left_linear_function,
            self._generate_triangular_function,
            self._generate_right_linear_function
        ]
        self._ranges = [
            (0, 1.5), (0.5, 2.0, 2.5), (9.0, 10.0)
        ]

        self._action_ranges = [
            (-0.5, 0.5), (0.5, 1.5), (1.5, 2.5),
            (2.5, 3.5), (3.5, 4.5), (4.5, 5.5),
            (5.5, 6.5), (6.5, 7.5)
        ]

    @classmethod
    def to_genotype(cls, phenotype):
        state = []
        for p in phenotype:
            if p == 'O':
                state.append('1.0')
            elif p == '.':
                state.append('0.0')
            else:
                state.append('9.0')
        return tuple(state)

    def to_membership_function(self, obs):
        obs = list(map(float, obs))
        memberships_values = [[] for _ in range(4)]
        for idx, _ in enumerate(obs[::2]):
            o = obs[idx * 2] + obs[idx * 2 + 1]
            for func, rang in zip(self._functions, self._ranges):
                memberships_values[idx].append(func(o, rang))
        return tuple(memberships_values)

    def calculate_final_actions_func_shape(self, values):
        final_ranges = []
        for value, action_range in zip(values, self._action_ranges):
            if not value:
                final_ranges.append([action_range[0], 0])
                final_ranges.append([action_range[1], 0])
                continue
            middle = (action_range[1] + action_range[0]) / 2.
            if value == 1:
                final_ranges.append([action_range[0], 0])
                final_ranges.append([middle, 1])
                final_ranges.append([action_range[1], 0])
            diff = middle - action_range[0]
            final_ranges.append([action_range[0], 0])
            final_ranges.append([middle - (value * diff), value])
            final_ranges.append([middle + (value * diff), value])
            final_ranges.append([action_range[1], 0])
        return final_ranges
