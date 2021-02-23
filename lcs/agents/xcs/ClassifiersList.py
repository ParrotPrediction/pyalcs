import numpy as np
import random
import copy

from lcs import TypedList, Perception
from lcs.agents.xcs import Classifier, Condition, Configuration


class ClassifiersList(TypedList):
    # TODO include cfg in the ClassifierList
    def __init__(self,
                 cfg: Configuration,
                 *args,
                 oktypes=(Classifier,),
                 ) -> None:
        self.cfg = cfg
        super().__init__(*args, oktypes=oktypes)

    def insert_in_population(self, cl: Classifier):
        for c in self:
            if c == cl:
                c.numerosity += 1
                return None
        self.append(cl)

    def generate_covering_classifier(self, situation, action, time_stamp):
        # both Perception and string has __getitem__
        # this way allows situation to be either str or Perception
        generalized = []
        for i in range(len(situation)):
            if np.random.rand() > self.cfg.population_wildcard:
                generalized.append(self.cfg.classifier_wildcard)
            else:
                generalized.append(situation[i])
        cl = Classifier(cfg=self.cfg,
                        condition=Condition(situation),
                        action=action,
                        time_stamp=time_stamp)
        return cl

    # Roulette-Wheel Deletion
    def delete_from_population(self):
        total_numerosity = sum(cl.numerosity for cl in self)
        if total_numerosity <= self.cfg.n:
            return None

        total_fitness = sum(cl.get_fitness() for cl in self)
        average_fitness = total_fitness / total_numerosity

        total_votes = 0
        deletion_votes = []

        for cl in range(len(self)):
            vote = self[cl].action_set_size * self[cl].numerosity
            sufficient_experience = (
                self[cl].experience > self.cfg.theta_del
            )
            low_fitness = (
                self[cl].get_fitness() / self[cl].numerosity <
                self.cfg.delta * average_fitness
            )
            if sufficient_experience and low_fitness:
                vote *= average_fitness / (self[cl].get_fitness /
                                           self[cl].numerosity)
            deletion_votes.append(vote)
            total_votes += vote

        selector = random.uniform(0, total_votes)
        for cl, vote in zip(self, deletion_votes):
            selector -= vote
            if selector <= 0:
                assert cl in self
                if self.safe_remove(cl):
                    return [cl]
                else:
                    return []

    def form_match_set(self, situation: Perception,  time_stamp):
        matching_ls = [cl for cl in self if cl.does_match(situation)]
        while len(matching_ls) < self.cfg.theta_mna:
            cl = self.generate_covering_classifier(situation, 1, time_stamp)
            self.insert_in_population(cl)
            self.delete_from_population()
            matching_ls.append(cl)
        return ClassifiersList(self.cfg, *matching_ls)

    def form_action_set(self, action):
        action_ls = [cl for cl in self if cl.action == action]
        return ClassifiersList(self.cfg, *action_ls)

