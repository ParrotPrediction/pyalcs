import numpy as np
import random
import logging

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
        for c in self:
            if c.condition == cl.condition and c.action == cl.action:
                c.numerosity += 1
                return None
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
        cl = Classifier(cfg=self.cfg,
                        condition=Condition(situation),
                        action=action,
                        time_stamp=time_stamp)
        return cl

    # Roulette-Wheel Deletion
    # TODO: use strategies
    def delete_from_population(self):
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
                cl.fitness / cl.numerosity > \
                self.cfg.delta * average_fitness:
            vote *= average_fitness / (cl.get_fitness() / cl.numerosity)
        return vote

    # I toyed with numerosity -= 1
    # and had better results with same solution as hosford42
    def _remove_based_on_votes(self, deletion_votes, selector):
        for cl, vote in zip(self, deletion_votes):
            selector -= vote
            if selector <= 0:
                assert cl in self
                self.safe_remove(cl)

    def form_match_set(self, situation: Perception,  time_stamp):
        matching_ls = [cl for cl in self if cl.does_match(situation)]
        while len(matching_ls) < self.cfg.number_of_actions:
            action = 0
            for a in range(0, self.cfg.number_of_actions):
                if all(cl.action != a for cl in matching_ls):
                    action = a
            cl = self.generate_covering_classifier(situation, action, time_stamp)
            self.insert_in_population(cl)
            self.delete_from_population()
            matching_ls.append(cl)
        return ClassifiersList(self.cfg, *matching_ls)

    def form_action_set(self, action):
        action_ls = [cl for cl in self if cl.action == action]
        return ClassifiersList(self.cfg, *action_ls)

    @property
    def numerosity(self):
        return sum(cl.numerosity for cl in self)

    # it is my creation, it very likely is wrong
    # reasoning: fitness is used as prediction weight in prediction array
    def best_prediction(self):
        return max(cl.prediction * cl.fitness for cl in self)
