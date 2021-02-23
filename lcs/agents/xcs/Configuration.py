import numpy as np
from lcs.strategies.action_selection import EpsilonGreedy


# TODO: Find if env should be here only in XCS
class Configuration(object):
    def __init__(self,
                 number_of_actions: int,
                 theta_mna: int,
                 classifier_wildcard: str = '#',
                 n: int = 200,
                 beta: float = 0.1,
                 alpha: float = 0.1,
                 epsilon_0: float = 10,
                 v: int = 5,
                 gamma: float = 0.71,
                 theta_ga: int = 25,
                 chi: float = 0.5,
                 mu: float = 0.01,
                 theta_del: int = 20,
                 delta: float = 0.1,
                 theta_sub: int = 20,
                 population_wildcard: float = 0.33,
                 p_i: float = float(np.finfo(np.float32).tiny),
                 epsilon_i: float = float(np.finfo(np.float32).tiny),
                 f_i: float = float(np.finfo(np.float32).tiny),
                 p_exp: float = 0.5,
                 do_ga_subsumption: bool = False,
                 do_action_set_subsumption: bool = False
                 ) -> None:
        """
        :param classifier_wildcard: Wildcard symbol
        :param n: maximum size of the population
        :param beta: learning rate for p, epsilon, f
        :param alpha: used in calculating fitness
        :param epsilon_0: used in calculating fitness
        :param v: power parameter, used in calculating fitness
        :param gamma: discount factor
        :param theta_ga: GA threshold, GA is applied when time since last GA is greater than  theta_GA
        :param chi: probability of applying crossover in the GA
        :param mu: probability of mutating an allele in the offspring
        :param theta_del: deletion threshold, after exp reaches theta_del fitness is probability for deletion
        :param delta: fraction of the mean fitness in P
        :param theta_sub: subsumption threshold, exp greater than theta_sub to be able to subsume
        :param population_wildcard: probability of using wildcard in one attribute in C
        :param p_i: used as initial value for new classifiers - prediction
        :param epsilon_i: used as initial value for new classifiers - error
        :param f_i: used as initial value for new classifiers - fitness
        :param p_exp: probability of choosing action uniform randomly
        :param theta_mna: minimal number of actions in match_set
        :param do_ga_subsumption: specifies if offspring are to be tested for logical subsumption
        :param do_action_set_subsumption: specifies if action sets are to be tested for subsuming classifiers
        """
        self.classifier_wildcard = classifier_wildcard
        self.n = n
        self.beta = beta
        self.alpha = alpha
        self.epsilon_0 = epsilon_0
        self.v = v
        self.gamma = gamma
        self.theta_GA = theta_ga
        self.chi = chi
        self.mu = mu
        self.theta_del = theta_del
        self.delta = delta
        self.theta_sub = theta_sub
        self.population_wildcard = population_wildcard
        self.p_i = p_i
        self.epsilon_i = epsilon_i
        self.f_i = f_i
        self.p_exp = p_exp
        self.theta_mna = theta_mna
        self.do_GA_subsumption = do_ga_subsumption
        self.do_action_set_subsumption = do_action_set_subsumption
        self.number_of_actions = number_of_actions

    def __str__(self) -> str:
        return str(vars(self))

    def initial_classifier_values(self):
        return self.p_i, self.epsilon_i, self.f_i


