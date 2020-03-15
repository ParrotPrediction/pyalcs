from typing import Callable

import lcs.agents.acs as acs
from lcs.agents import EnvironmentAdapter
from lcs.strategies.action_selection import EpsilonGreedy


class Configuration(acs.Configuration):
    def __init__(self,
                 classifier_length: int,
                 number_of_possible_actions: int,
                 classifier_wildcard: str = '#',
                 environment_adapter=EnvironmentAdapter,
                 user_metrics_collector_fcn: Callable = None,
                 fitness_fcn=None,
                 metrics_trial_frequency: int = 5,
                 do_pee: bool = False,
                 do_ga: bool = False,
                 do_subsumption: bool = True,
                 do_action_planning: bool = False,
                 action_planning_frequency: int = 50,
                 beta: float = 0.05,
                 gamma: float = 0.95,
                 theta_i: float = 0.1,
                 theta_r: float = 0.9,
                 initial_q: float = 0.5,
                 action_selector=EpsilonGreedy,
                 epsilon: float = 0.5,
                 biased_exploration_prob: float = 0.05,
                 u_max: int = 100000,
                 theta_exp: int = 20,
                 theta_ga: int = 100,
                 theta_as: int = 20,
                 mu: float = 0.3,
                 chi: float = 0.8):

        super(Configuration, self).__init__(
            classifier_length,
            number_of_possible_actions,
            classifier_wildcard,
            environment_adapter,
            user_metrics_collector_fcn,
            fitness_fcn,
            metrics_trial_frequency,
            do_subsumption,
            beta,
            theta_i,
            theta_r,
            epsilon,
            u_max,
            theta_exp,
            theta_as)

        self.gamma = gamma
        self.do_pee = do_pee
        self.do_ga = do_ga
        self.do_action_planning = do_action_planning
        self.initial_q = initial_q
        self.action_planning_frequency = action_planning_frequency
        self.action_selector = action_selector(
            all_actions=number_of_possible_actions,
            epsilon=epsilon,
            biased_exploration_prob=biased_exploration_prob
        )
        self.biased_exploration_prob = biased_exploration_prob
        self.theta_ga = theta_ga
        self.mu = mu
        self.chi = chi

    def __str__(self) -> str:
        return str(vars(self))

