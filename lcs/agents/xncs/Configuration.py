import numpy as np
from typing import Callable

import lcs.agents.xcs as xcs
from lcs.agents import EnvironmentAdapter


class Configuration(xcs.Configuration):

    def __init__(self,
                 number_of_actions: int,
                 lmc: int = 100,
                 lem: float = 1,

                 # theta_mna it is actually smart to make it equal to number of actions
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
        self.lmc = lmc
        self.lem = lem
        super().__init__(
            number_of_actions=number_of_actions,
            classifier_wildcard=classifier_wildcard,
            environment_adapter=environment_adapter,
            max_population=max_population,  # n
            learning_rate=learning_rate,  # beta
            alpha=alpha,
            epsilon_0=epsilon_0,
            v=v,  # nu
            gamma=gamma,
            ga_threshold=ga_threshold,
            chi=chi,
            mutation_chance=mutation_chance,  # mu
            deletion_threshold=deletion_threshold,  # theta_del
            delta=delta,
            subsumption_threshold=subsumption_threshold,  # theta_sub
            covering_wildcard_chance=covering_wildcard_chance,  # population wildcard
            initial_prediction=initial_prediction,  # p_i
            initial_error=initial_error,  # epsilon_i
            initial_fitness=initial_fitness,  # f_i
            epsilon=epsilon,  # p_exp, exploration probability
            do_ga_subsumption=do_ga_subsumption,
            do_action_set_subsumption=do_action_set_subsumption,
            metrics_trial_frequency=metrics_trial_frequency,
            user_metrics_collector_fcn=user_metrics_collector_fcn,
            multistep_enfiroment=multistep_enfiroment
        )

