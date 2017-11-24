
class ACS2Configuration:
    def __init__(self,
                 classifier_length,
                 number_of_possible_actions,
                 classifier_wildcard='#',
                 perception_mapper_fcn=None,
                 action_mapping_dict=None,
                 environment_metrics_fcn=None,
                 performance_fcn=None,
                 performance_fcn_params={},
                 do_ga=False,
                 do_subsumption=True,
                 beta=0.05,
                 gamma=0.95,
                 theta_i=0.1,
                 theta_r=0.9,
                 epsilon=0.5,
                 u_max=100000,
                 theta_exp=20,
                 thera_ga=100,
                 theta_as=20,
                 mu=0.3,
                 chi=0.8):
        """
        Creates the configuration object used during training the ACS2 agent.

        :param classifier_length: length of the condition and effect strings
        :param number_of_possible_actions: number of possible actions to
            be executed
        :param classifier_wildcard: wildcard symbol
        :param perception_mapper_fcn:
        :param action_mapping_dict: dictionary where key is internal ID of
            action (numbers from `0 .. number_of_possible_actions)` and value
            is the environmental representation of the action.
        :param environment_metrics_fcn:
        :param performance_fcn: function for estimating agent performance
        :param performance_fcn_params: optional parameters needed for
            calculating agent performance
        :param do_ga: switch *Genetic Generalization* module
        :param do_subsumption:
        :param beta:
        :param gamma:
        :param theta_i:
        :param theta_r:
        :param epsilon:
        :param u_max:
        :param theta_exp:
        :param thera_ga:
        :param theta_as:
        :param mu:
        :param chi:
        """
        if performance_fcn_params is None:
            performance_fcn_params = {}
        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.classifier_wildcard = classifier_wildcard
        self.perception_mapper_fcn = perception_mapper_fcn
        self.action_mapping_dict = action_mapping_dict
        self.environment_metrics_fcn = environment_metrics_fcn
        self.performance_fcn = performance_fcn
        self.performance_fcn_params = performance_fcn_params
        self.do_ga = do_ga
        self.do_subsumption = do_subsumption
        self.theta_exp = theta_exp
        self.beta = beta
        self.gamma = gamma
        self.theta_i = theta_i
        self.theta_r = theta_r
        self.epsilon = epsilon
        self.u_max = u_max
        self.theta_ga = thera_ga
        self.theta_as = theta_as
        self.mu = mu
        self.chi = chi

    def __str__(self):
        return "ACS2Configuration:" \
               "\n\t- Classifier length: [{}]" \
               "\n\t- Number of possible actions: [{}]" \
               "\n\t- Classifier wildcard: [{}]" \
               "\n\t- Perception mapper function: [{}]" \
               "\n\t- Action mapping dict: [{}]" \
               "\n\t- Environment metrics function: [{}]" \
               "\n\t- Performance calculation function: [{}] " \
               "\n\t- Do GA: [{}]" \
               "\n\t- Do subsumption: [{}]" \
               "\n\t- Beta: [{}]" \
               "\n\t- ..." \
               "\n\t- Epsilon: [{}]" \
               "\n\t- U_max: [{}]" \
            .format(self.classifier_length,
                    self.number_of_possible_actions,
                    self.classifier_wildcard,
                    self.perception_mapper_fcn,
                    self.action_mapping_dict,
                    self.environment_metrics_fcn,
                    self.performance_fcn,
                    self.do_ga,
                    self.do_subsumption,
                    self.beta,
                    self.epsilon,
                    self.u_max)
