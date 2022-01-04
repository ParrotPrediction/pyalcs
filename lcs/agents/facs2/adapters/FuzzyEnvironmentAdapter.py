import gym
import numpy as np


class FuzzyEnvironmentAdapter(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return observation

    @classmethod
    def _generate_triangular_function(cls,
                                      x: float,
                                      abc: [float]) -> float:
        """
        Generate traingular membership function and
        calcaulate value for given variable.

        Parameters
        ----------
        x: float
            given position
        abc: [float]
            values to describe shape of triangle

        Returns
        -------
        float
            value of given point
        """
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
        """
        Change 2D list of memberships value to 1D

        Parameters
        ----------
        raw_state
            current state of environment

        Returns
        -------
        [float]
            1D list of memberships values
        """
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
        """
        Generate left linear function to calculate
        membership value of given x

        Parameters
        ----------
        x: float
            given position
        ab: [float]
            values to describe shape of function

        Returns
        -------
        float
            value of given point
        """

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
        """
        Generate right linear function to calculate
        membership value of given x

        Parameters
        ----------
        x: float
            given position
        ab: [float]
            values to describe shape of function

        Returns
        -------
        float
            value of given point
        """
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
        """
        Calculate centroid coordinates for final action function

        Parameters
        ----------
        ranges
            list of points describing shape of action func

        Returns
        -------
        float, float
            Coordinates x and y of calculated centroid
        """
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
        """
        Calculate final shape of action function

        Parameters
        ----------
        values
            calculated max value for each possible action
            from all classifiers

        Returns
        -------
        [float, float]
            coordinates for every point describing
             function shape

        """
        raise NotImplementedError()

    def to_membership_function(self, obs):
        """
        Change given observation to membership values

        Parameters
        ----------
        obs
            observations from environment

        Returns
        -------
        [float]
             list of membership values
        """
        raise NotImplementedError()
