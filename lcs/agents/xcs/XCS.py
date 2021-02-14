from dataclasses import dataclass
from typing import Union, Optional, Generator, List, Dict
from typing import Callable, List, Tuple
import numpy as np

from lcs.agents import Agent
from lcs.agents.xcs import Configuration, Classifier, ClassifiersList
from lcs.agents.Agent import TrialMetrics
from lcs.agents.ImmutableSequence import ImmutableSequence


# The condition C is included inside the ImmutableSequence
# Find proper object description for reinforcement program
class XCS(Agent):
    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None
                 ) -> None:
        """
        :param cfg: place to store most variables
        :param population: all classifiers at current time
        """
        self.cfg = cfg
        self.population = population

    def get_population(self):
        return self.cfg

    def get_cfg(self):
        return self.population

    # run experiment
    # TODO: return Trial Metrics
    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        situation = env.to_genotype()
        match_set = self.population.form_match_set(situation)
        prediction_array = self._generate_prediction_array(match_set)
        action = self.select_action(prediction_array, match_set)
        # stopped at line 6 of Run_Experiment
        raise NotImplementedError

    # Functions for XCS
    @classmethod
    def _generate_prediction_array(cls, match_set: ClassifiersList):
        prediction_array = []
        fitness_sum_array = []
        for cl in match_set:
            prediction_array.append(cl.prediction())
            fitness_sum_array.append(cl.fitness())
        for i in range(0, len(prediction_array)):
            if fitness_sum_array[i] != 0:
                prediction_array[i] /= fitness_sum_array[i]
        return prediction_array

    # TODO: from lcs.strategies.action_selection import EpsilonGreedy
    def select_action(self, prediction_array, match_set: ClassifiersList) -> int:
        if np.random.rand() > self.cfg.p_exp:
            return match_set[prediction_array.index(max(prediction_array))].action
        return np.random.randint(match_set).action
