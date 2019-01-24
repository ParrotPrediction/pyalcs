from typing import Tuple

import numpy as np


class RealValueEncoder:
    r"""
    Real-value encoder.

    Assumes that provided values are already in range [0,1].

    This min-max normalization can be done as a prior step using the following
    formula:

    .. math::
        x\prime = \frac{x - \textrm{min}(x)} {\textrm{max}(x)-\textrm{min}(x)}
    """
    def __init__(self, resolution_bits: int) -> None:
        resolution = pow(2, resolution_bits)
        step = 1 / resolution
        self.upper_max = resolution - 1
        self.splits = [x * step for x in range(0, resolution + 1)]

    @property
    def range(self) -> Tuple[int, int]:
        """
        Range the range (min,max) of available encoded values.

        Returns
        -------
        Tuple[int, int]
            Min, max values
        """
        return 0, self.upper_max

    def encode(self, val: float, noise: float = 0.0) -> int:
        """
        Encodes the float value into `[0, 2^bits]` states.
        This results in `2^bits + 1` different states.

        Parameters
        ----------
        val : float
            real-valued number in range [0,1]
        noise: float
            noise that is appended to the perception

        Returns
        -------
        int
            discrete state within resolution
        """
        if val < 0 or val > 1:
            raise ValueError("Value not in correct [0, 1] range")

        # Disturb value and limit it within range
        val = np.clip(val + noise, 0, 1)

        # Check boundary conditions
        if val == 0:
            return self.range[0]

        if val == 1:
            return self.range[1]

        # In other cases iterate over splits to find correct bucket
        bucket = -1
        for i, _ in enumerate(self.splits[:-1]):
            x1 = self.splits[i]
            x2 = self.splits[i + 1]

            if x1 <= val < x2:
                bucket = i

        assert bucket != -1

        return bucket

    def decode(self, encoded_val: int) -> float:
        """
        Decodes a discrete value to real-valued representation (still [0,1]
        range)

        Parameters
        ----------
        encoded_val : int
            encoded value

        Returns
        -------
        float
            real-valued number from [0,1] range

        """
        if encoded_val < 0 or encoded_val > self.upper_max:
            raise ValueError("Value is not from possible resolution range")

        return encoded_val / self.upper_max
