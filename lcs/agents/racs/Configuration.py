from typing import Callable

from lcs.agents import EnvironmentAdapter
from lcs.representations import Interval, FULL_INTERVAL


class Configuration:
    def __init__(self,
                 classifier_length: int,
                 number_of_possible_actions: int,
                 environment_adapter=EnvironmentAdapter,
                 user_metrics_collector_fcn: Callable=None,
                 metrics_trial_frequency: int = 5,
                 do_ga: bool=False,
                 do_subsumption: bool=True,
                 beta: float=0.05,
                 gamma: float=0.95,
                 theta_i: float=0.1,
                 theta_r: float=0.9,
                 epsilon: float=0.5,
                 cover_noise: float = 0.1,
                 mutation_noise: float = 0.1,
                 u_max: int=100000,
                 theta_exp: int=20,
                 theta_ga: int=100,
                 theta_as: int=20,
                 mu: float=0.3,
                 chi: float=0.8) -> None:

        self.oktypes = (Interval,)

        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.classifier_wildcard = FULL_INTERVAL

        self.environment_adapter = environment_adapter

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
