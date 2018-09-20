from lcs.representations import UBR
from lcs.representations.RealValueEncoder import RealValueEncoder


class Configuration:
    def __init__(self,
                 classifier_length: int,
                 number_of_possible_actions: int,
                 encoder_bits: int,
                 perception_mapper_fcn=None,
                 action_mapping_fcn=None,
                 do_ga=False,
                 beta=0.05,
                 gamma=0.95,
                 theta_i=0.1,
                 theta_r=0.9,
                 epsilon=0.5,
                 u_max=100000,
                 theta_exp=20,
                 theta_ga=100,
                 theta_as=20) -> None:

        self.oktypes = (UBR,)
        self.encoder = RealValueEncoder(encoder_bits)

        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.classifier_wildcard = UBR(*self.encoder.range)

        self.perception_mapper_fcn = perception_mapper_fcn
        self.action_mapping_fcn = action_mapping_fcn
        self.do_ga = do_ga

        self.beta = beta
        self.gamma = gamma
        self.theta_i = theta_i
        self.theta_r = theta_r
        self.epsilon = epsilon
        self.u_max = u_max

        self.theta_exp = theta_exp
        self.theta_ga = theta_ga
        self.theta_as = theta_as
