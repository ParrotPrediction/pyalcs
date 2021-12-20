from lcs.agents.facs2.adapters.FuzzyEnvironmentAdapter import \
    FuzzyEnvironmentAdapter


class MountainCarFuzzyEnvironmentAdapter(FuzzyEnvironmentAdapter):
    _position_min = -1.2
    _position_max = 0.6
    _velocity_min = -0.07
    _velocity_max = 0.07
    condition_length = 9

    def __init__(self):
        self._position_functions = [
            self._generate_left_linear_function,
            self._generate_triangular_function,
            self._generate_triangular_function,
            self._generate_triangular_function,
            self._generate_right_linear_function]

        self._velocity_functions = [
            self._generate_left_linear_function,
            self._generate_triangular_function,
            self._generate_triangular_function,
            self._generate_right_linear_function]

        self._position_ranges = [
            (self._position_min, -1), (-1.1, -0.8, -0.5), (-0.6, -0.3, 0),
            (-0.1, 0.2, 0.5), (0.4, self._position_max)
        ]
        self._velocity_ranges = [
            (self._velocity_min, -0.04), (-0.05, -0.02, 0.01),
            (-0.01, 0.02, 0.05), (0.04, self._velocity_max)
        ]
        self._action_ranges = [
            (-1.0, 1.0),
            (0.0, 2.0),
            (1.0, 3.0)
        ]

    @classmethod
    def to_genotype(cls, phenotype):
        state = []
        for p in phenotype:
            state.append(str(p))
        return tuple(state)

    def to_membership_function(self, obs):
        position = float(obs[0])
        velocity = float(obs[1])
        membership_function_values = [[], []]
        for pos_func, pos_range in zip(self._position_functions,
                                       self._position_ranges):
            membership_function_values[0].append(pos_func(position,
                                                              pos_range))

        for vel_func, vel_range in zip(self._velocity_functions,
                                       self._velocity_ranges):
            membership_function_values[1].append(vel_func(velocity,
                                                              vel_range))

        return tuple(membership_function_values)

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
