from lcs.agents import Agent
from lcs.agents.xcs import Configuration
from lcs.agents.Agent import TrialMetrics


class Classifier:
    def __init__(self):
        raise NotImplementedError()


class ClassifierList:
    def __init__(self):
        raise NotImplementedError()

    def generate_match_set(self):
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
        self.match_set = None
        self.action_set = None
        self.action_set_previous = None
        raise NotImplementedError()

    def get_population(self):
        return self.cfg

    def get_cfg(self):
        return self.population

    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        raise NotImplementedError()

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        raise NotImplementedError()
