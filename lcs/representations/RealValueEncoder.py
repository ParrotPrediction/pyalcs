class RealValueEncoder:
    """
    Assumes that provided values are in range [0,1].

    This min-max normalization can be done as a prior step using the following
    formula:

    x' = (x - min(x)) / (max(x) - min(x))
    """
    def __init__(self, resolution_bits: int):
        self.resolution = pow(2, resolution_bits)

    def encode(self, val: float):
        """
        Encodes the float value into [0, 2^bits] states.
        This results in 2^bits + 1 different states

        :param val: real-valued number in range [0,1]
        :return: discrete state within resolution
        """
        if val < 0 or val > 1:
            raise ValueError("Value is not normalized within [0,1] range")

        return int(val * self.resolution)

    def decode(self, encoded_val: int) -> float:
        """
        Decodes a discrete value to real-valued representation (still [0,1]
        range)

        :param encoded_val: encoded value
        :return: real-valued number from [0,1] range
        """
        if encoded_val < 0 or encoded_val > self.resolution:
            raise ValueError("Value is not from possible resolution range")

        return encoded_val / self.resolution
