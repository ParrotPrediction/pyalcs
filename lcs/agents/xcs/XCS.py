import numpy as np
from typing import Optional

from lcs.agents import Agent
from lcs.agents.xcs import Configuration, Classifier, ClassifiersList
from lcs.agents.Agent import TrialMetrics
from lcs.strategies.action_selection import EpsilonGreedy

# TODO: Find proper object description for reinforcement program
# TODO: Logger


class XCS(Agent):
    def __init__(self,
                 cfg: Configuration,
                 population: Optional[ClassifiersList] = None
                 ) -> None:
        """
        :param cfg: place to store most variables
        :param population: all classifiers at current time
        """
        self.cfg = cfg
        if population is not None:
            self.population = population
        else:
            self.population = ClassifiersList(cfg=cfg)
        self.time_stamp = 0

    def get_population(self):
        return self.cfg

    def get_cfg(self):
        return self.population

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        self.cfg.p_exp = 0
        self._run_trial_explore(env, trials, current_trial)

    # run experiment
    # TODO: return Trial Metrics
    # TODO: make it compatible with Open AI environments
    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
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
            # TODO: I am using to_phenotype to determine eop and reward
            # I ma not sure if it is right.
            reward, eop = env.to_phenotype(action)
            if len(prev_action_set) > 0:
                p = prev_reward + self.cfg.gamma * max(prediction_array)
                self._update_set(prev_action_set, p)
                self.run_ga(prev_situation)
            if eop:
                p = reward
                self._update_set(prev_action_set, p)
                self.run_ga(prev_situation)
            else:
                prev_action_set = action_set
                prev_reward = reward
                prev_situation = situation

            self.time_stamp += 1
        raise NotImplementedError

    # TODO: change it to ClassifierList method
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

    def _update_set(self, action_set: ClassifiersList, p):
        for cl in action_set:
            cl.expirience += 1
            action_set_numerosity = sum(cl.numerosity for cl in action_set)
            # update prediction, prediction error, action set size estimate
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

    def update_fitness(self, action_set: ClassifiersList):
        accuracy_sum = 0
        # It literally is called that in paper
        accuracy_vector_k = []
        for cl in action_set:
            if cl.error < self.cfg.epsilon_i:
                tmp_acc = 1
            else:
                tmp_acc = self.cfg.alpha * pow(cl.error * self.cfg.epsilon_i, self.cfg.v)
            accuracy_vector_k.append(tmp_acc)
            accuracy_sum += tmp_acc + cl.numerosity
        for cl, k in zip(action_set, accuracy_vector_k):
            cl.fitness += self.cfg.beta * (k * cl.numerosity / accuracy_sum - cl.fitness)

    def do_action_set_subsumption(self, action_set: ClassifiersList) -> None:
        cl = None
        for c in action_set:
            if c.could_subsume():
                if cl is None or c.more_general(cl):
                    cl = c
        if cl is not None:
            for c in action_set:
                if cl.is_more_general(c):
                    cl.numerosity += c.numerosity
                    action_set.safe_remove(c)
                    self.population.safe_remove(c)

    # TODO: Implement run GA
    def run_ga(self, situation):
        raise NotImplementedError()
