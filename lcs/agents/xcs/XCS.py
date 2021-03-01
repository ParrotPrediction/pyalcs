import logging
import numpy as np
import copy
from typing import Optional

from lcs.agents import Agent
from lcs.agents.xcs import Configuration, Classifier, ClassifiersList
from lcs.agents.Agent import TrialMetrics
from lcs.strategies.action_selection import EpsilonGreedy

# TODO: Find proper object description for reinforcement program
# TODO: Logger

logger = logging.getLogger(__name__)

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

    def get_population(self):
        return self.cfg

    def get_cfg(self):
        return self.population

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        self.cfg.p_exp = 0
        return self._run_trial_explore(env, trials, current_trial)

    # run experiment
    # TODO: Realize what trials, current_trial do
    # TODO: make it compatible with Open AI environments
    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        eop = False
        prev_action_set = None
        prev_reward = None
        prev_situation = None
        time_stamp = 0  # steps
        done = False  # eop

        raw_state = env.reset()
        # situation is called state in ACS
        situation = self.cfg.environment_adapter.to_genotype(raw_state)

        while not done:
            match_set = self.population.form_match_set(situation, time_stamp)
            prediction_array = self.generate_prediction_array(match_set)
            action = self.select_action(prediction_array, match_set)
            action_set = match_set.form_action_set(action)
            # TODO: I am using to_phenotype to determine eop and reward
            # I ma not sure if it is right.
            raw_state, reward, done, _ = env.step(action)
            situation = self.cfg.environment_adapter.to_genotype(raw_state)
            if len(prev_action_set) > 0:
                p = prev_reward + self.cfg.gamma * max(prediction_array)
                self._update_set(prev_action_set, p)
                self.run_ga(prev_action_set, prev_situation)
            if eop:
                p = reward
                self._update_set(prev_action_set, p)
                self.run_ga(action_set, situation)
            else:
                prev_action_set = action_set
                prev_reward = reward
                prev_situation = situation
            time_stamp += 1
        return TrialMetrics(time_stamp, prev_reward)

    def generate_prediction_array(self, match_set: ClassifiersList):
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
            cl.experience += 1
            action_set_numerosity = sum(cl.numerosity for cl in action_set)
            # update prediction, prediction error, action set size estimate
            if cl.experience < 1/self.cfg.beta:
                cl.prediction += (p - cl.prediction) / cl.experience
                cl.error += (abs(p - cl.prediction) - cl.error) / cl.experience
                cl.action_set_size +=\
                    (action_set_numerosity - cl.action_set_size) / cl.experience
            else:
                cl.prediction += self.cfg.beta * (p - cl.prediction)
                cl.error += self.cfg.beta * (abs(p - cl.prediction) - cl.error)
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

    def run_ga(self, action_set, situation, time_stamp):
        if time_stamp - sum(cl.time_stamp * cl.numerosity for cl in action_set) / \
                sum(cl.numerosity for cl in action_set) > self.cfg.theta_GA:
            for i in enumerate(action_set):
                action_set[i].time_stamp = time_stamp
            parent1 = self.select_offspring(action_set)
            parent2 = self.select_offspring(action_set)
            child1 = copy.copy(parent1)
            child2 = copy.copy(parent2)
            child1.numerosity = child2.numerosity + 1
            child1.experience = child2.experience + 1
            if np.random.rand() < self.cfg.chi:
                self.apply_crossover(child1, child2)
                child1.prediction = (parent1.prediction + parent2.prediction) / 2
                child1.error = 0.25 * (parent1.error + parent2.error) / 2
                child1.fitness = (parent1.fitness + parent2.fitness) / 2
                child2.prediction = child1.prediction
                child2.error = child1.error
                child2.fitness = child2.fitness
            self.apply_mutation(child1, situation)
            self.apply_mutation(child2, situation)
            if self.cfg.do_GA_subsumption:
                if parent1.does_subsume(child1):
                    parent1.numerosity += 1
                elif parent2.does_subsume(child1):
                    parent2.numerosity += 1
                else:
                    self.population.insert_in_population(child1)
                self.population.delete_from_population()

                if parent1.does_subsume(child2):
                    parent1.numerosity += 1
                elif parent2.does_subsume(child2):
                    parent2.numerosity += 1
                else:
                    self.population.insert_in_population(child2)
                self.population.delete_from_population()

    def select_offspring(self, action_set: ClassifiersList) -> Classifier:
        fitness_sum = 0
        for cl in action_set:
            fitness_sum += cl.fitness
        choice_point = np.random.rand() * fitness_sum
        fitness_sum = 0
        for cl in action_set:
            fitness_sum += cl.fitness
            if fitness_sum > choice_point:
                return cl

    def apply_crossover(self, child1, child2):
        x = np.random.rand() * len(child1.condition)
        y = np.random.rand() * len(child1.condition)
        if x > y:
            x, y = y, x
        i = 0
        while i < y:
            if x <= i < y:
                child1.condition[i], child2.condition[i] =\
                    child1.condition[i], child2.condition[i]
            i += 1

    def apply_mutation(self, child, situation):
        i = 0
        while i < len(child.condition):
            if np.random.rand() < self.cfg.mu:
                if child.condition[i] == child.condition.WILDCARD:
                    child.condition[i] = situation[i]
                else:
                    child.condition[i] = child.condition.WILDCARD
            i += 1
        if np.random.rand() < self.cfg.mu:
            child.action = np.random.randint(self.cfg.number_of_actions)
