from lcs.agents.facs2.adapters.FuzzyEnvironmentAdapter import \
    FuzzyEnvironmentAdapter
import numpy as np


class CartPoleFuzzyEnvironmentAdapter(FuzzyEnvironmentAdapter):
    _cart_position_min = -4.8
    _cart_position_max = 4.8
    _cart_velocity_min = -np.inf
    _cart_velocity_max = np.inf
    _pole_angle_min = -0.418
    _pole_angle_max = 0.418
    _pole_angular_velocity_min = -np.inf
    _pole_angular_velocity_max = np.inf

    condition_length = 18

    def __init__(self):
        self._position_functions = [
            self._generate_left_linear_function,
            self._generate_triangular_function,
            self._generate_triangular_function,
            self._generate_triangular_function,
            self._generate_right_linear_function
        ]

        self._velocity_functions = [
            self._generate_left_linear_function,
            self._generate_triangular_function,
            self._generate_right_linear_function
        ]

        self._pole_angle_functions = [
            self._generate_left_linear_function,
            self._generate_triangular_function,
            self._generate_triangular_function,
            self._generate_triangular_function,
            self._generate_triangular_function,
            self._generate_triangular_function,
            self._generate_right_linear_function
        ]

        self._angular_velocity_functions = [
            self._generate_left_linear_function,
            self._generate_triangular_function,
            self._generate_right_linear_function
        ]

        self._position_ranges = [
            (self._cart_position_min, -2.4), (self._cart_position_min, -2.4, 0),
            (-2.4, 0, 2.4), (0, 2.4, self._cart_position_max),
            (2.4, self._cart_position_max)
        ]

        self._velocity_ranges = [
            (-1, 0), (-1, 0, 1), (0, 1)
        ]

        self._pole_angle_ranges = [
            (-0.418, -0.279), (-0.418, -0.279, -0.139), (-0.279, -0.139, 0),
            (-0.139, 0, 0.139), (0, 0.139, 0.279),
            (0.139, 0.279, 0.418), (0.279, 0.418)
        ]

        self._angular_velocity_ranges = [
            (-2, 0), (-2, 0, 2), (0, 2)
        ]

        self._action_ranges = [
            (-1.0, 1.0),
            (0.0, 2.0)
        ]

    @classmethod
    def to_genotype(cls, phenotype):
        state = []
        for p in phenotype:
            state.append(str(p))
        return tuple(state)

    def to_membership_function(self, obs):
        cart_position = float(obs[0])
        cart_velocity = float(obs[1])
        pole_angle = float(obs[2])
        pole_angular_velocity = float(obs[3])
        membership_function_values = [[], [], [], []]
        for pos_func, pos_range in zip(self._position_functions,
                                       self._position_ranges):
            membership_function_values[0].append(pos_func(cart_position,
                                                          pos_range))
        for vel_func, vel_range in zip(self._velocity_functions,
                                       self._velocity_ranges):
            membership_function_values[1].append(vel_func(cart_velocity,
                                                          vel_range))

        for pole_angle_func, angle_range in zip(self._pole_angle_functions,
                                                self._pole_angle_ranges):
            membership_function_values[2].append(pole_angle_func(pole_angle,
                                                                 angle_range))

        for angular_vel_func, pole_angular_velocity_range in zip(
            self._angular_velocity_functions, self._angular_velocity_ranges):
            membership_function_values[3].append(angular_vel_func(
                pole_angular_velocity, pole_angular_velocity_range))

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
