from lcs.representations import UBR
from lcs.representations.RealValueEncoder import RealValueEncoder


class Configuration:
    def __init__(self,
                 classifier_length: int,
                 number_of_possible_actions: int,
                 encoder_bits: int,
                 beta=0.05,
                 u_max=100000,) -> None:

        self.oktypes = (UBR,)
        self.encoder = RealValueEncoder(encoder_bits)

        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.classifier_wildcard = UBR(*self.encoder.range)

        self.beta = beta
        self.u_max = u_max
