import logging
import numpy as np
from copy import copy
from typing import Optional

from lcs.agents import Agent
from lcs.agents.xcs import Configuration, Classifier, ClassifiersList
from lcs.agents.Agent import TrialMetrics
from lcs.strategies.reinforcement_learning import simple_q_learning
from lcs.strategies.action_selection import EpsilonGreedy


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
        self.act_reward = [0 for _ in range(cfg.number_of_actions)]
        self.time_stamp = 0
        self.done = False

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        temp = self.cfg.epsilon
        self.cfg.epsilon = 0
        metrics = self._run_trial_explore(env, trials, current_trial)
        self.cfg.epsilon = temp
        return metrics

    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        prev_action_set = None
        prev_reward = None
        prev_state = None  # state is known as situation
        self.time_stamp = 0  # steps
        self.done = False  # eop
        reward = None

        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)

        while not self.done:
            self.population.delete_from_population()
            # We are in t+1 here
            match_set = self.population.form_match_set(state, self.time_stamp)
            prediction_array = self.generate_prediction_array(match_set)
            action = self.select_action(prediction_array, match_set)
            action_set = match_set.form_action_set(action)
            # apply action to environment
            raw_state, step_reward, done, _ = env.step(action)
            state = self.cfg.environment_adapter.to_genotype(raw_state)
            reward = simple_q_learning(self.act_reward[action],
                                       step_reward,
                                       self.cfg.learning_rate,
                                       self.cfg.gamma,
                                       match_set.best_prediction())

            self._distribute_and_update(prev_action_set, prev_state, prev_reward, prediction_array)
            if self.done:
                # we won't be able to do t next loop round so we do it now
                p = reward
                self.update_set(prev_action_set, p)
                self.run_ga(action_set, state, self.time_stamp)
            # moving values from t+1 to t
            prev_action_set = copy(action_set)
            prev_reward = copy(reward)
            prev_state = copy(state)
            self.time_stamp += 1
        return TrialMetrics(self.time_stamp, reward)

    # TODO: Test it
    def _distribute_and_update(self, action_set, situation, reward, prediction_array):
        if action_set is not None and len(action_set) > 0:
            p = reward + self.cfg.gamma * max(prediction_array)
            self.update_set(action_set, p)
            self.run_ga(action_set, situation, self.time_stamp)

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
        if np.random.rand() > self.cfg.epsilon:
            return match_set[prediction_array.index(max(prediction_array))].action
        return np.random.randint(self.cfg.number_of_actions)

    def update_set(self, action_set: ClassifiersList, p):
        if action_set is not None and len(action_set) > 0:
            for cl in action_set:
                cl.experience += 1
                action_set_numerosity = sum(cl.numerosity for cl in action_set)
                # update prediction, prediction error, action set size estimate
                if cl.experience < 1/self.cfg.learning_rate:
                    cl.prediction += (p - cl.prediction) / cl.experience
                    cl.error += (abs(p - cl.prediction) - cl.error) / cl.experience
                    cl.action_set_size +=\
                        (action_set_numerosity - cl.action_set_size) / cl.experience
                else:
                    cl.prediction += self.cfg.learning_rate * (p - cl.prediction)
                    cl.error += self.cfg.learning_rate * (abs(p - cl.prediction) - cl.error)
                    cl.action_set_size += \
                        self.cfg.learning_rate * (action_set_numerosity - cl.action_set_size)
            self.update_fitness(action_set)
            if self.cfg.do_action_set_subsumption:
                self.do_action_set_subsumption(action_set)

    def update_fitness(self, action_set: ClassifiersList):
        accuracy_sum = 0
        accuracy_vector_k = []
        for cl in action_set:
            if cl.error < self.cfg.epsilon_0:
                tmp_acc = 1
            else:
                tmp_acc = self.cfg.alpha * pow(1/(cl.error * self.cfg.epsilon_0), self.cfg.v)
            accuracy_vector_k.append(tmp_acc)
            accuracy_sum += tmp_acc + cl.numerosity
        for cl, k in zip(action_set, accuracy_vector_k):
            cl.fitness += self.cfg.learning_rate * (k * cl.numerosity / accuracy_sum - cl.fitness)

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
        if action_set is None:
            return None

        temp_numerosity = sum(cl.numerosity for cl in action_set)
        if temp_numerosity == 0:
            return None

        if time_stamp - sum(cl.time_stamp * cl.numerosity for cl in action_set) / temp_numerosity \
            > self.cfg.ga_threshold:
            for cl in action_set:
                cl.time_stamp = time_stamp
            # select children
            parent1 = self.select_offspring(action_set)
            parent2 = self.select_offspring(action_set)
            child1 = copy(parent1)
            child2 = copy(parent2)
            child1.numerosity = 1
            child2.numerosity = 1
            child1.experience = 0
            child2.experience = 0
            # apply crossover
            if np.random.rand() < self.cfg.chi:
                self.apply_crossover(child1, child2)
                child1.prediction = (parent1.prediction + parent2.prediction) / 2
                child1.error = 0.25 * (parent1.error + parent2.error) / 2
                child1.fitness = (parent1.fitness + parent2.fitness) / 2
                child2.prediction = child1.prediction
                child2.error = child1.error
                child2.fitness = child1.fitness
            # apply mutation on both children
            self.apply_mutation(child1, situation)
            self.apply_mutation(child2, situation)
            # apply subsumption or just insert into population
            if self.cfg.do_GA_subsumption:
                if parent1.does_subsume(child1):
                    parent1.numerosity += 1
                elif parent2.does_subsume(child1):
                    parent2.numerosity += 1
                else:
                    self.population.insert_in_population(child1)

                if parent1.does_subsume(child2):
                    parent1.numerosity += 1
                elif parent2.does_subsume(child2):
                    parent2.numerosity += 1
                else:
                    self.population.insert_in_population(child2)

            else:
                self.population.insert_in_population(child1)
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
            if np.random.rand() < self.cfg.mutation_chance:
                if child.condition[i] == child.condition.WILDCARD:
                    child.condition[i] = situation[i]
                else:
                    child.condition[i] = child.condition.WILDCARD
            i += 1
        if np.random.rand() < self.cfg.mutation_chance:
            child.action = np.random.randint(self.cfg.number_of_actions)
