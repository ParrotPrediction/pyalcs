from typing import Callable

from lcs.representations import UBR


class Configuration:
    def __init__(self,
                 classifier_length: int,
                 number_of_possible_actions: int,
                 encoder=None,
                 perception_mapper_fcn=None,
                 action_mapping_fcn=None,
                 user_metrics_collector_fcn: Callable=None,
                 metrics_trial_frequency: int = 5,
                 do_ga=False,
                 do_subsumption=True,
                 beta=0.05,
                 gamma=0.95,
                 theta_i=0.1,
                 theta_r=0.9,
                 epsilon=0.5,
                 cover_noise: float = 0.1,
                 mutation_noise: float = 0.1,
                 u_max=100000,
                 theta_exp=20,
                 theta_ga=100,
                 theta_as=20,
                 mu=0.3,
                 chi=0.8) -> None:

        if encoder is None:
            raise TypeError('Real number encoder should be passed')

        self.oktypes = (UBR,)
        self.encoder = encoder

        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.classifier_wildcard = UBR(*self.encoder.range)

        self.perception_mapper_fcn = perception_mapper_fcn
        self.action_mapping_fcn = action_mapping_fcn

        self.metrics_trial_frequency = metrics_trial_frequency
        self.user_metrics_collector_fcn = user_metrics_collector_fcn

        self.do_ga = do_ga
        self.do_subsumption = do_subsumption

        self.beta = beta
        self.gamma = gamma
        self.theta_i = theta_i
        self.theta_r = theta_r
        self.epsilon = epsilon
        # Max range of uniform noise distribution that can alter
        # the perception during covering U[0, cover_noise]
        self.cover_noise = cover_noise

        # Max range of uniform noise distribution that can broaden the
        # phenotype interval range.
        self.mutation_noise = mutation_noise
        self.u_max = u_max

        self.theta_exp = theta_exp
        self.theta_ga = theta_ga
        self.theta_as = theta_as

        self.mu = mu
        self.chi = chi
