from dataclasses import dataclass
from typing import Union, Optional, Generator, List, Dict
from lcs import TypedList, Perception

from lcs.agents import Agent
from lcs.agents.xcs import Configuration
from lcs.agents.Agent import TrialMetrics
from lcs.agents.ImmutableSequence import ImmutableSequence


class ClassifierList:
    def __init__(self):
        raise NotImplementedError()


# The condition C is included inside the ImmutableSequence
# The action A should be included inside EnvironmentalAdapter INSIDE THE CONFIGURATION
# Find proper object description for reinforcement program
class XCS(Agent):
    def __init__(self,
                 cfg: Configuration,
                 population: ClassifierList = None
                 ) -> None:
        """
        :param cfg: place to store most variables
        :param population: all classifiers at current time
        """
        self.cfg = cfg
        self.population = population
        raise NotImplementedError()

        self.eop = False

    def get_population(self):
        return self.cfg

    def get_cfg(self):
        return self.population

    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        raise NotImplementedError()

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        situation = self.get_situation()
        match_set = self.generate_match_set(self.population, situation)
        prediction_array = self.generate_prediction_array(match_set)
        action = self.select_action(prediction_array)
        action_set = self.generate_action_set(match_set, action)
        # execute said actions here
        # get reward
        # dalej są if

        raise NotImplementedError()

    def get_situation(self, env):
        raise NotImplementedError()

    def generate_match_set(self, population, situation):
        raise NotImplementedError()

    def generate_prediction_array(self, match_set):
        raise NotImplementedError()

    def select_action(self, prediction_array):
        raise NotImplementedError()

    def generate_action_set(self, match_set, action):
        raise NotImplementedError()
