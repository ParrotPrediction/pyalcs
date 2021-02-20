import numpy as np

from lcs.agents import Agent
from lcs.agents.xcs import Configuration, Classifier, ClassifiersList
from lcs.agents.Agent import TrialMetrics

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
        self.population = ClassifiersList(cfg=cfg)
        self.time_stamp = 0

    def get_population(self):
        return self.cfg

    def get_cfg(self):
        return self.population

    # run experiment
    # TODO: return Trial Metrics
    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        reward = 0
        eop = False
        prev_action_set = None
        prev_reward = 0
        prev_situation = None

        while not eop:
            situation = env.to_genotype()
            match_set = self.population.form_match_set(situation, self.time_stamp)
            prediction_array = self._generate_prediction_array(match_set)
            action = self.select_action(prediction_array, match_set)
            action_set = match_set.form_action_set(action)
            # TODO: See if this is correct way to do it.
            # I expect reinforcement program to be on the other side of adapted
            reward = env.to_phenotype(action)
            # TODO: insert GA here
            if len(prev_action_set) > 0:
                p = prev_reward + self.cfg.gamma
                # TODO: Change it so action set will have pointers to population cls
                self._update_set(prev_action_set, p)



            # TODO: eop flag is not changed
            if self.cfg.eop:
                pass
            else:
                prev_action_set = action_set
                prev_reward = reward
                prev_situation = situation

            self.time_stamp += 1
        raise NotImplementedError

    @classmethod
    def _generate_prediction_array(cls, match_set: ClassifiersList):
        prediction_array = []
        fitness_sum_array = []
        for cl in match_set:
            prediction_array.append(cl.prediction())
            fitness_sum_array.append(cl.get_fitness())
        for i in range(0, len(prediction_array)):
            if fitness_sum_array[i] != 0:
                prediction_array[i] /= fitness_sum_array[i]
        return prediction_array

    # TODO: YOu can use EpsilonGreedy
    def select_action(self, prediction_array, match_set: ClassifiersList) -> int:
        if np.random.rand() > self.cfg.p_exp:
            return match_set[prediction_array.index(max(prediction_array))].action
        return np.random.randint(match_set).action

    # TODO: Update Set
    def _update_set(self, p):
        raise NotImplementedError()

    # TODO: Run GA
    def _run_genetic_algorithm(self):
        raise NotImplementedError()
