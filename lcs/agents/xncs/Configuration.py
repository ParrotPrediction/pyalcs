import numpy as np
from typing import Callable

import lcs.agents.xcs as xcs


class Configuration(xcs.Configuration):

    def __init__(self,
                 number_of_actions: int,  # theta_mna it is actually smart to make it equal to number of actions
                 lmc: int = 100,
                 lem: float = 1,
                 classifier_wildcard: str = '#',
                 max_population: int = 200,  # n
                 learning_rate: float = 0.1,  # beta
                 alpha: float = 0.1,
                 epsilon_0: float = 10,
                 v: int = 5,
                 gamma: float = 0.71,
                 ga_threshold: int = 25,
                 chi: float = 0.5,
                 mutation_chance: float = 0.01,  # mu
                 deletion_threshold: int = 20,  # theta_del
                 delta: float = 0.1,
                 subsumption_threshold: int = 20,  # theta_sub
                 covering_wildcard_chance: float = 0.33,  # population wildcard
                 initial_prediction: float = float(np.finfo(np.float32).tiny),  # p_i
                 initial_error: float = float(np.finfo(np.float32).tiny),  # epsilon_i
                 initial_fitness: float = float(np.finfo(np.float32).tiny),  # f_i
                 epsilon: float = 0.5,  # p_exp, exploration probability
                 do_ga_subsumption: bool = False,
                 do_action_set_subsumption: bool = False,
                 metrics_trial_frequency: int = 5,
                 user_metrics_collector_fcn: Callable = None
                 ) -> None:
        self.lmc = lmc
        self.lem = lem
        self.classifier_wildcard = classifier_wildcard
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


