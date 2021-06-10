import numpy as np
from typing import Callable

from lcs.agents import EnvironmentAdapter


class Configuration(object):
    def __init__(self,
                 number_of_actions: int,  # theta_mna it is actually smart to make it equal to number of actions
                 classifier_wildcard: str = '#',
                 environment_adapter=EnvironmentAdapter,
                 max_population: int = 200,  # n
                 learning_rate: float = 0.1,  # beta
                 alpha: float = 0.1,
                 epsilon_0: float = 10,
                 v: int = 5,  # nu
                 gamma: float = 0.71,
                 ga_threshold: int = 25,
                 chi: float = 0.5,
                 mutation_chance: float = 0.01,  # mu
                 deletion_threshold: int = 20,  # theta_del
                 delta: float = 0.1,
                 subsumption_threshold: int = 20,  # theta_sub
                 covering_wildcard_chance: float = 0.33,  # population wildcard
                 initial_prediction: float = 0.000001,  # p_i
                 initial_error: float = 0.000001,  # epsilon_i
                 initial_fitness: float = 0.000001,  # f_i
                 epsilon: float = 0.5,  # p_exp, exploration probability
                 do_ga_subsumption: bool = False,
                 do_action_set_subsumption: bool = False,
                 metrics_trial_frequency: int = 5,
                 user_metrics_collector_fcn: Callable = None,
                 multistep_enfiroment: bool = True
                 ) -> None:
        """
        :param classifier_wildcard: Wildcard symbol
        :param max_population: maximum size of the population
        :param learning_rate: learning rate for p, epsilon, f
        :param alpha: used in calculating fitness
        :param epsilon_0: used in calculating fitness
        :param v: power parameter, used in calculating fitness
        :param gamma: discount factor
        :param ga_threshold: GA threshold, GA is applied when time since last GA is greater than  ga_threshold
        :param chi: probability of applying crossover in the GA
        :param mutation_chance: probability of mutating an allele in the offspring
        :param deletion_threshold: deletion threshold, after exp reaches deletion_threshold fitness is probability for deletion
        :param delta: fraction of the mean fitness in P
        :param subsumption_threshold: subsumption threshold, exp greater than subsumption_threshold to be able to subsume
        :param covering_wildcard_chance: probability of using wildcard in one attribute in C
        :param initial_prediction: used as initial value for new classifiers - prediction
        :param initial_error: used as initial value for new classifiers - error
        :param initial_fitness: used as initial value for new classifiers - fitness
        :param epsilon: probability of choosing action uniform randomly
        :param number_of_actions: minimal number of actions in match_set
        :param do_ga_subsumption: specifies if offspring are to be tested for logical subsumption
        :param do_action_set_subsumption: specifies if action sets are to be tested for subsuming classifiers
        """
        self.classifier_wildcard = classifier_wildcard
        self.environment_adapter = environment_adapter
        self.max_population = max_population
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon_0 = epsilon_0
        self.v = v
        self.gamma = gamma
        self.ga_threshold = ga_threshold
        self.chi = chi
        self.mutation_chance = mutation_chance
        self.deletion_threshold = deletion_threshold
        self.delta = delta
        self.subsumption_threshold = subsumption_threshold
        self.covering_wildcard_chance = covering_wildcard_chance
        self.initial_prediction = initial_prediction
        self.initial_error = initial_error
        self.initial_fitness = initial_fitness
        self.epsilon = epsilon  # p_exp, probability of exploration
        self.number_of_actions = number_of_actions
        self.do_GA_subsumption = do_ga_subsumption
        self.do_action_set_subsumption = do_action_set_subsumption
        self.metrics_trial_frequency = metrics_trial_frequency
        self.user_metrics_collector_fcn = user_metrics_collector_fcn
        self.multistep_enfiroment = multistep_enfiroment

    def __str__(self) -> str:
        return str(vars(self))


