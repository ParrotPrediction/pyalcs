from __future__ import annotations

import logging
import random
from collections import deque
from typing import Union, Optional

from enum import Enum
from lcs import TypedList, Perception
from lcs.agents import ImmutableSequence, Agent
from lcs.agents.Agent import TrialMetrics

logging.basicConfig(level=logging.DEBUG)


class ClassifierTrace(Enum):
    BAD = 0
    GOOD = 1


class Configuration:
    def __init__(self,
                 classifier_length: int,
                 number_of_possible_actions: int,
                 feature_possible_values: list,
                 trace_length: int = 5):
        assert classifier_length == len(feature_possible_values)
        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.feature_possible_values = feature_possible_values
        self.trace_length = trace_length


class Condition(ImmutableSequence):
    def does_match(self, p: Perception) -> bool:
        for ci, oi in zip(self, p):
            if ci != self.WILDCARD and oi != self.WILDCARD and ci != oi:
                return False

        return True

    def subsumes(self, other) -> bool:
        raise NotImplementedError('YACS has no subsume operator')


class Effect(ImmutableSequence):
    @staticmethod
    def diff(p0: Perception, p1: Perception):
        # Computes the desired effect
        return Effect(
            [p1i if p1i != p0i else ImmutableSequence.WILDCARD for p0i, p1i in
             zip(p0, p1)])

    def subsumes(self, other) -> bool:
        raise NotImplementedError('YACS has no subsume operator')


class Classifier:
    def __init__(self,
                 condition: Union[Condition, str, None] = None,
                 action: Optional[int] = None,
                 effect: Union[Effect, str, None] = None,
                 cfg: Optional[Configuration] = None):

        if cfg is None:
            raise TypeError("Configuration should be passed to Classifier")

        self.cfg = cfg

        def build_perception_string(cls, initial,
                                    length=self.cfg.classifier_length):
            if initial:
                return cls(initial)

            return cls.empty(length=length)

        self.condition = build_perception_string(Condition, condition)
        self.action = action
        self.effect = build_perception_string(Effect, effect)
        self.trace = deque(maxlen=self.cfg.trace_length)

    def does_match(self, situation: Perception) -> bool:
        return self.condition.does_match(situation)

    def add_to_trace(self, mark: ClassifierTrace):
        self.trace.append(mark)


class ClassifiersList(TypedList):
    def __init__(self, *args, oktypes=(Classifier,)) -> None:
        super().__init__(*args, oktypes=oktypes)

    def form_match_set(self, situation: Perception) -> ClassifiersList:
        matching = [cl for cl in self if cl.does_match(situation)]
        return ClassifiersList(*matching)

    def form_action_set(self, action: int) -> ClassifiersList:
        matching = [cl for cl in self if cl.action == action]
        return ClassifiersList(*matching)


class LatentLearning:
    """
    The process is in charge of discovering C-A-E classifiers with maximally
    general C parts that accurately model the dynamics of the environment.
    """

    def __init__(self, cfg: Configuration):
        self.cfg = cfg

    def effect_covering(self,
                        population: ClassifiersList,
                        p0: Perception,
                        p1: Perception,
                        prev_action: int):

        # Calculate the desired effect
        de = Effect.diff(p0, p1)

        # Every classifier from the previous action set is analyzed
        match_set = population.form_match_set(p0)
        action_set = match_set.form_action_set(prev_action)

        # Add trace to classifiers that anticipated correctly
        good_classifiers = [cl for cl in action_set if cl.effect == de]
        for cl in good_classifiers:
            cl.add_to_trace(ClassifierTrace.GOOD)

        # Add trace to classifiers that anticipated wrong
        wrong_classifiers = [cl for cl in action_set if cl.effect != de]
        for cl in [cl for cl in action_set if cl.effect != de]:
            cl.add_to_trace(ClassifierTrace.BAD)

        # If no classifier has correct anticipation - create it
        if len(good_classifiers) == 0:
            old_cl = random.choice(wrong_classifiers)
            new_cl = Classifier(
                condition=Condition(old_cl.condition),
                action=old_cl.action,
                effect=de,
                cfg=self.cfg
            )
            new_cl.add_to_trace(ClassifierTrace.GOOD)

            population.append(new_cl)

    def select_accurate_classifiers(self, population: ClassifiersList):
        for cl in population:
            if len(cl.trace) == self.cfg.trace_length:
                if not all(t == ClassifierTrace.GOOD for t in cl.trace):
                    population.remove(cl)


class YACS(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None):
        self.cfg = cfg
        self.population = population or ClassifiersList()

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        logging.info("Running trial explore")

        prev_state = Perception.empty()
        prev_action = None

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        logging.info("Running trial exploit")


if __name__ == '__main__':
    cfg = Configuration(4, 2)
    agent = YACS(cfg)
