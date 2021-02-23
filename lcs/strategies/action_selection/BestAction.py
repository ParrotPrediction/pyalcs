import random

import lcs.agents.acs as acs
import lcs.agents.acs2 as acs2
import lcs.agents.xcs as xcs
from . import RandomAction


class BestAction:

    def __init__(self, all_actions: int):
        self.all_actions = all_actions

    def __call__(self, population):
        if type(population) is acs.ClassifiersList:
            return self._handle_best_action_for_acs(population)
        elif type(population) is acs2.ClassifiersList:
            return self._handle_best_action_for_acs2(population)
        elif type(population) is xcs.ClassifiersList:
            return self._handle_best_action_for_xcs(population)
        else:
            raise TypeError()

    def _handle_best_action_for_acs(self, population) -> int:
        best_classifier = max(population, key=lambda cl: cl.fitness)
        return best_classifier.action

    def _handle_best_action_for_acs2(self, population) -> int:
        anticipated_change_cls = [cl for cl in population
                                  if cl.does_anticipate_change()]

        if len(anticipated_change_cls) > 0:
            random.shuffle(anticipated_change_cls)
            best_classifier = max(anticipated_change_cls,
                                  key=lambda cl: cl.fitness * cl.num)

            if best_classifier is not None:
                return best_classifier.action

        # If there is no classifier - return random action
        return RandomAction(all_actions=self.all_actions)(population)

    def _handle_best_action_for_xcs(self, population) -> int:
        best_classifier = max(population, key=lambda cl: cl.fitness)
        return best_classifier.action
