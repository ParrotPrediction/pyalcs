import numpy as np
import random
from dataclasses import dataclass
from typing import Union, Optional, Generator, List, Dict
from typing import Callable, List, Tuple

from lcs import TypedList, Perception
from lcs.agents.xcs import Classifier, Condition


class ClassifiersList(TypedList):
    # TODO include cfg in the ClassifierList
    def __init__(self, *args, oktypes=(Classifier,)) -> None:
        super().__init__(*args, oktypes=oktypes)

    def form_match_set(self, situation: Perception, cfg, time_stamp):
        matching_ls = [cl for cl in self if cl.does_match(situation)]
        while matching_ls.count() < cfg.minimum_actions:
            cl = self.generate_covering_classifier(situation, time_stamp, cfg)
            self.insert_in_population(cl)
            self.delete_from_population(cfg)
            matching_ls.append(cl)
        return ClassifiersList(*matching_ls)

    def form_action_set(self, action):
        action_ls = [cl for cl in self if cl.action == action]
        return action_ls

    def insert_in_population(self, cl):
        for c in self:
            if c.condition == cl.condition and c.action == cl.action:
                c.numerosity += 1
            return None
        self.append(cl)

    def generate_covering_classifier(self, situation, time_stamp, cfg):
        # TODO Test Condition and fix situation typing
        cl = Classifier(condition=Condition(situation),
                        action=random.randint(0, cfg.number_of_actions),
                        time_stamp=time_stamp)
        for i in enumerate(cl.condition):
            if np.random.rand() > cfg.population_wildcard:
                cl.condition[i] = cfg.classifier_wildcard

        return cl

    def delete_from_population(self, cfg):
        if len(self) <= cfg.n:
            return None
        total_numerosity = sum(cl.numerosity for cl in self)
        total_fitness = sum(cl.fitness for cl in self)
        average_fitness = total_fitness / total_numerosity

        total_votes = 0
        deletion_votes = {}

        for cl in self:
            vote = cl.action_set_size * cl.numerosity
            sufficient_experience = (
                cl.experience > cfg.theta_del
            )
            low_fitness = (
                cl.fitness / cl.numerosity <
                cfg.delta * average_fitness
            )
            if sufficient_experience and low_fitness:
                vote *= average_fitness / (cl.fitness /
                                           cl.numerosity)

            deletion_votes[cl] = vote
            total_votes += vote

        selector = random.uniform(0, total_votes)
        for rule, vote in deletion_votes.items():
            selector -= vote
            if selector <= 0:
                assert rule in self
                if self.safe_remove(rule):
                    return [rule]
                else:
                    return []

