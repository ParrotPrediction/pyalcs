import numpy as np

from lcs.agents import Agent
from lcs.agents.xcs import Configuration, Classifier, ClassifiersList
from lcs.agents.Agent import TrialMetrics
from lcs.strategies.action_selection import EpsilonGreedy

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

    # Make it so it runs with p_exp always choosing exploit
    # TODO: change p_exp in cfg and run it.
    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        raise NotImplementedError()

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
            prediction_array = self.generate_prediction_array(match_set)
            action = self.select_action(prediction_array, match_set)
            action_set = match_set.form_action_set(action)
            # TODO: See if this is correct way to do it.
            # I expect reinforcement program to be on the other side of adapter
            reward = env.to_phenotype(action)
            # Code continues from line 9
            # TODO: insert GA here
            if len(prev_action_set) > 0:
                p = prev_reward + self.cfg.gamma
                # TODO: Change it so action set will have pointers to population cls
                self._update_set(prev_action_set, p)

            # TODO: eop flag is not changed
            if eop:
                pass
            else:
                prev_action_set = action_set
                prev_reward = reward
                prev_situation = situation

            self.time_stamp += 1
        raise NotImplementedError

    @classmethod
    def generate_prediction_array(cls, match_set: ClassifiersList):
        prediction_array = []
        fitness_sum_array = []
        for cl in match_set:
            prediction_array.append(cl.prediction * cl.fitness)
            fitness_sum_array.append(cl.fitness)
        for i in range(0, len(prediction_array)):
            if fitness_sum_array[i] != 0:
                prediction_array[i] /= fitness_sum_array[i]
        return prediction_array

    # TODO: EspilonGreedy
    # Run into a lot of issues where in EpsilonGreedy where BestAction was not callable
    # Changing EpsilonGreed to:
    # best = BestAction(all_actions=self.all_actions)
    # return best(population)
    # Fixed the issue but I want to solve it without changes to EpsilonGreedy.py
    def select_action(self, prediction_array, match_set: ClassifiersList) -> int:
        if np.random.rand() > self.cfg.p_exp:
            return match_set[prediction_array.index(max(prediction_array))].action
        return match_set[np.random.randint(len(match_set))].action

    def _update_set(self, action_set, p):
        for cl in action_set:
            cl.raise_expirienc()
            action_set_numerosity = sum(cl.numerosity for cl in action_set)
            # update prediction
            # update prediction error
            # update action set size estimate
            if cl.expirience < 1/self.cfg.beta:
                cl.prediction += (p - cl.prediction) / cl.expirience
                cl.error += (abs(p - cl.prediction) - p.error) / cl.expirience
                cl.action_set_size +=\
                    (action_set_numerosity - cl.action_set_size) / cl.expirience
            else:
                cl.prediction += self.cfg.beta * (p - cl.prediction)
                cl.error += self.cfg.beta * (abs(p - cl.prediction) - p.error)
                cl.action_set_size += \
                    self.cfg.beta * (action_set_numerosity - cl.action_set_size)
        self.update_fitness(action_set)
        if self.cfg.do_action_set_subsumption:
            self.do_action_set_subsumption(action_set)

    # TODO: Update Fitness
    def update_fitness(self, action_set):
        raise NotImplementedError()

    # TODO: this method
    def do_action_set_subsumption(self, action_set):
        raise NotImplementedError()
