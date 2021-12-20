from lcs.agents.EnvironmentAdapter import EnvironmentAdapter
import numpy as np


class FuzzyEnvironmentAdapter(EnvironmentAdapter):

    @classmethod
    def _generate_triangular_function(cls,
                                      x: float,
                                      abc: [float]) -> float:
        assert len(abc) == 3
        a, b, c = np.r_[abc]
        assert a <= b <= c

        if x < a or x > c:
            return 0.0

        if x > b:
            y = (c - x) / (c - b)
        elif x < b:
            y = (x - a) / (b - a)
        else:
            y = 1.0

        return y

    def change_state_type(self, raw_state):
        final_state = []
        state = self.to_membership_function(raw_state)
        for obs in state:
            for o in obs:
                final_state.append(str(o))
        return final_state

    @classmethod
    def _generate_left_linear_function(cls,
                                       x: float,
                                       ab: [float]) -> float:
        assert len(ab) == 2
        a, b = np.r_[ab]
        assert a <= b

        if x <= a:
            y = 1.0
        elif a < x < b:
            y = (b - x) / (b - a)
        else:
            y = 0.0

        return y

    @classmethod
    def _generate_right_linear_function(cls,
                                        x: float,
                                        ab: [float]) -> float:
        assert len(ab) == 2
        a, b = np.r_[ab]
        assert a <= b

        if x >= b:
            y = 1.0
        elif a < x < b:
            y = (b - x) / (b - a)
        else:
            y = 0.0

        return y

    def calculate_centroid(self, ranges):
        a = cx = cy = 0

        for i, (x, y) in enumerate(ranges):
            if i == len(ranges) - 1:
                break
            next_xy = ranges[i + 1]
            cx += (x + next_xy[0]) * (x * next_xy[1] - next_xy[0] * y)
            cy += (y + next_xy[1]) * (x * next_xy[1] - next_xy[0] * y)
            a += (x * next_xy[1] - next_xy[0] * y)
        a /= 2
        cx /= 6 * a
        cy /= 6 * a
        return cx, cy

    def calculate_final_actions_func_shape(self, values):
        raise NotImplementedError()

    def to_membership_function(self, obs):
        raise NotImplementedError()
