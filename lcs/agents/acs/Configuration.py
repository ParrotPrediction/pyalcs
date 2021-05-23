from typing import Callable

from lcs.agents import EnvironmentAdapter


class Configuration(object):
    def __init__(self,
                 classifier_length: int,
                 number_of_possible_actions: int,
                 classifier_wildcard: str = '#',
                 environment_adapter=EnvironmentAdapter,
                 user_metrics_collector_fcn: Callable = None,
                 fitness_fcn=None,
                 metrics_trial_frequency: int = 5,
                 model_checkpoint_frequency: int = None,
                 do_subsumption: bool = True,
                 beta: float = 0.05,
                 # gamma: float = 0.95,
                 theta_i: float = 0.1,
                 theta_r: float = 0.9,
                 epsilon: float = 0.5,
                 u_max: int = 100000,
                 theta_exp: int = 20,
                 theta_as: int = 20,
                 use_mlflow: bool = False) -> None:
        """
        Creates the configuration object used during training the ACS2 agent.

        :param classifier_length: length of the condition and effect strings
        :param number_of_possible_actions: number of possible actions to
            be executed
        :param classifier_wildcard: wildcard symbol
        :param environment_adapter: EnvironmentAdapter class ACS2 needs to use
            to interact with the environment
        :param fitness_fcn: Custom fitness function
        :param do_subsumption:
        :param beta:
        :param theta_i: inadequacy threshold
        :param theta_r: float
            Reliability threshold. Quality level when the classifier is
            treated as "reliable"
        :param epsilon: float
            Probability of executing random action. Otherwise the action
            from best classifier is selected.
        :param u_max:
        :param theta_exp:
        :param theta_as:
        """
        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.classifier_wildcard = classifier_wildcard
        self.environment_adapter = environment_adapter
        self.metrics_trial_frequency = metrics_trial_frequency
        self.model_checkpoint_freq = model_checkpoint_frequency
        self.user_metrics_collector_fcn = user_metrics_collector_fcn
        self.fitness_fcn = fitness_fcn
        self.do_subsumption = do_subsumption
        self.theta_exp = theta_exp
        self.beta = beta
        self.theta_i = theta_i
        self.theta_r = theta_r
        self.epsilon = epsilon
        self.u_max = u_max
        self.theta_as = theta_as
        self.use_mlflow = use_mlflow

    def __str__(self) -> str:
        return str(vars(self))
