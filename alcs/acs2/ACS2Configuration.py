
class ACS2Configuration:
    def __init__(self, classifier_length, number_of_possible_actions,
                 classifier_wildcard='#', beta=0.05, gamma=0.95, theta_i=0.1,
                 theta_r=0.9, epsilon=1.0, u_max=100000, theta_exp=20,
                 thera_ga=100, theta_as=20, mu=0.3, chi=0.8):
        self.classifier_length = classifier_length
        self.number_of_posiible_actions = number_of_possible_actions
        self.classifier_wildcard = classifier_wildcard
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


default_configuration = ACS2Configuration(
    classifier_length=8, number_of_possible_actions=8)
