from lcs.representations.RealValueEncoder import RealValueEncoder


class UBR:
    """
    Real-value representation for unordered-bounded values.
    """
    def __init__(self, resolution_bits):
        self.encoder = RealValueEncoder(resolution_bits)
        self.x1, self.x2 = None, None  # bounds
