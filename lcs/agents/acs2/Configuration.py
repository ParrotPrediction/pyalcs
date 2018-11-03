from typing import Callable

from lcs.agents import EnvironmentAdapter


class Configuration:
    def __init__(self,
                 classifier_length: int,
                 number_of_possible_actions: int,
                 classifier_wildcard='#',
                 environment_adapter=EnvironmentAdapter,
                 user_metrics_collector_fcn: Callable = None,
                 metrics_trial_frequency: int = 5,
                 do_ga: bool=False,
                 do_subsumption: bool=True,
                 do_action_planning: bool=False,
                 action_planning_frequency: int=50,
                 beta: float=0.05,
                 gamma: float=0.95,
                 theta_i: float=0.1,
                 theta_r: float=0.9,
                 epsilon: float=0.5,
                 u_max: int=100000,
                 theta_exp: int=20,
                 theta_ga: int=100,
                 theta_as: int=20,
                 mu: float=0.3,
                 chi: float=0.8) -> None:
        """
        Creates the configuration object used during training the ACS2 agent.

        :param classifier_length: length of the condition and effect strings
        :param number_of_possible_actions: number of possible actions to
            be executed
        :param classifier_wildcard: wildcard symbol
        :param environment_adapter: EnvironmentAdapter class ACS2 needs to use
            to interact with the environment
        :param do_ga: switch *Genetic Generalization* module
        :param do_subsumption:
        :param do_action_planning: switch Action Planning phase
        :param action_planning_frequency:
        :param beta:
        :param gamma:
        :param theta_i: inadequacy threshold
        :param theta_r:
        :param epsilon:
        :param u_max:
        :param theta_exp:
        :param theta_as:
        :param theta_as:
        :param mu:
        :param chi: GA crossover probability
        """
        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.classifier_wildcard = classifier_wildcard
        self.environment_adapter = environment_adapter
        self.metrics_trial_frequency = metrics_trial_frequency
        self.user_metrics_collector_fcn = user_metrics_collector_fcn
        self.do_ga = do_ga
        self.do_subsumption = do_subsumption
        self.do_action_planning = do_action_planning
        self.action_planning_frequency = action_planning_frequency
        self.theta_exp = theta_exp
        self.beta = beta
        self.gamma = gamma
        self.theta_i = theta_i
        self.theta_r = theta_r
        self.epsilon = epsilon
        self.u_max = u_max
        self.theta_ga = theta_ga
        self.theta_as = theta_as
        self.mu = mu
        self.chi = chi

    def __str__(self):
        return "ACS2Configuration:" \
               "\n\t- Classifier length: [{}]" \
               "\n\t- Number of possible actions: [{}]" \
               "\n\t- Classifier wildcard: [{}]" \
               "\n\t- Environment adapter function: [{}]" \
               "\n\t- Do GA: [{}]" \
               "\n\t- Do subsumption: [{}]" \
               "\n\t- Do Action Planning: [{}]" \
               "\n\t- Beta: [{}]" \
               "\n\t- ..." \
               "\n\t- Epsilon: [{}]" \
               "\n\t- U_max: [{}]" \
            .format(self.classifier_length,
                    self.number_of_possible_actions,
                    self.classifier_wildcard,
                    self.environment_adapter,
                    self.do_ga,
                    self.do_subsumption,
                    self.do_action_planning,
                    self.beta,
                    self.epsilon,
                    self.u_max)
