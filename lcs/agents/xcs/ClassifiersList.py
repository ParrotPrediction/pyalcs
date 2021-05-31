import logging
import random

import numpy as np

from lcs import TypedList, Perception
from lcs.agents.xcs import Classifier, Condition, Configuration
logger = logging.getLogger(__name__)


class ClassifiersList(TypedList):
    def __init__(self,
                 cfg: Configuration,
                 *args,
                 oktypes=(Classifier,),
                 ) -> None:
        self.cfg = cfg
        super().__init__(*args, oktypes=oktypes)

    def insert_in_population(self, cl: Classifier):
        existing_classifiers = [c for c in self if c == cl]
        if len(existing_classifiers) > 0:
            assert len(existing_classifiers) == 1, 'duplicates found, while inserting'
            existing_classifiers[0].numerosity += 1
        else:
            self.append(cl)

    def generate_covering_classifier(self, situation, action, time_stamp):
        # both Perception and string has __getitem__
        # this way allows situation to be either str or Perception
        generalized = []
        for i in range(len(situation)):
            if np.random.rand() > self.cfg.covering_wildcard_chance:
                generalized.append(self.cfg.classifier_wildcard)
            else:
                generalized.append(situation[i])

        return Classifier(condition=Condition(generalized),
                          action=action,
                          time_stamp=time_stamp,
                          cfg=self.cfg)

    def _generate_covering_and_insert(self, situation, action, time_stamp):
        cl = self.generate_covering_classifier(situation, action, time_stamp)
        self.insert_in_population(cl)
        self.delete_from_population()
        return cl

    # Roulette-Wheel Deletion
    # TODO: use strategies
    def delete_from_population(self):
        # TODO: change while to if
        # there are places where more than one rule enters the population
        # to remedy it I just made deletion run until it cleared all of them
        # proffered method should be running it once, ideally inside
        # insert_into_population
        while self.numerosity > self.cfg.max_population:
            average_fitness = sum(cl.fitness for cl in self) / self.numerosity
            deletion_votes = []
            for cl in self:
                deletion_votes.append(self._deletion_vote(cl, average_fitness))
            selector = random.uniform(0, sum(deletion_votes))
            self._remove_based_on_votes(deletion_votes, selector)

    def _deletion_vote(self, cl, average_fitness):
        vote = cl.action_set_size * cl.numerosity
        if cl.experience > self.cfg.deletion_threshold and \
            cl.fitness / cl.numerosity < \
                self.cfg.delta * average_fitness:
            vote *= average_fitness / (cl.fitness / cl.numerosity)
        return vote

    def _remove_based_on_votes(self, deletion_votes, selector):
        for cl, vote in zip(self, deletion_votes):
            selector -= vote
            if selector <= 0:
                if cl.numerosity > 1:
                    cl.numerosity -= 1
                    return cl
                else:
                    self.safe_remove(cl)
                    return cl

    def generate_match_set(self, situation: Perception, time_stamp):
        matching_ls = [cl for cl in self if cl.does_match(situation)]
        action = self._find_not_present_action(matching_ls)
        while action is not None:
            cl = self._generate_covering_and_insert(situation, action, time_stamp)
            matching_ls.append(cl)
            action = self._find_not_present_action(matching_ls)
        return ClassifiersList(self.cfg, *matching_ls)

    def _find_not_present_action(self, matching_set):
        for action in range(0, self.cfg.number_of_actions):
            if all(cl.action != action for cl in matching_set):
                return action

    def generate_action_set(self, action):
        action_ls = [cl for cl in self if cl.action == action]
        return ClassifiersList(self.cfg, *action_ls)

    @property
    def numerosity(self):
        return sum(cl.numerosity for cl in self)

    @property
    def best_prediction(self):
        return max(cl.prediction * cl.fitness for cl in self)

    @property
    def prediction_array(self):
        prediction_array = [0 for _ in range(self.cfg.number_of_actions)]
        fitness_sum_array = [0 for _ in range(self.cfg.number_of_actions)]
        for cl in self:
            prediction_array[cl.action] += cl.prediction * cl.fitness
            fitness_sum_array[cl.action] += cl.fitness
        for i in range(self.cfg.number_of_actions):
            if fitness_sum_array[i] != 0:
                prediction_array[i] /= fitness_sum_array[i]
        return prediction_array

    def update_set(self, p):
        action_set_numerosity = sum(cl.numerosity for cl in self)
        for cl in self:
            cl.experience += 1
            # update prediction, prediction error, action set size estimate
            if cl.experience < 1 / self.cfg.learning_rate:
                cl.prediction += \
                    (p - cl.prediction) / cl.experience
                cl.error += \
                    (abs(p - cl.prediction) - cl.error) / cl.experience
                cl.action_set_size += \
                    (action_set_numerosity - cl.action_set_size) / cl.experience
            else:
                cl.prediction +=\
                    self.cfg.learning_rate * (p - cl.prediction)
                cl.error += \
                    self.cfg.learning_rate * (abs(p - cl.prediction) - cl.error)
                cl.action_set_size += \
                    self.cfg.learning_rate * (action_set_numerosity - cl.action_set_size)
        self._update_fitness()

    def _update_fitness(self):
        accuracy_sum = 0
        accuracy_vector_k = []
        for cl in self:
            if cl.error < self.cfg.epsilon_0:
                tmp_acc = 1
            else:
                tmp_acc = (pow(self.cfg.alpha * (cl.error * self.cfg.epsilon_0), -self.cfg.v))
            accuracy_vector_k.append(tmp_acc)
            accuracy_sum += tmp_acc + cl.numerosity
        for cl, k in zip(self, accuracy_vector_k):
            cl.fitness += (
                self.cfg.learning_rate *
                (k * cl.numerosity / accuracy_sum - cl.fitness)
            )


