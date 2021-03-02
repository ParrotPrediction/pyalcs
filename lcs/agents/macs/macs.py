from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Union, Optional

from lcs import TypedList, Perception
from lcs.agents import Agent, ImmutableSequence
from lcs.agents.Agent import TrialMetrics


@dataclass
class DontCare:
    symbol: str = '#'
    eis: float = 0.5  # expected improvement by specialization

    def __repr__(self):
        return self.symbol


class Condition(ImmutableSequence):
    WILDCARD = DontCare()
    OK_TYPES = (str, DontCare)

    def __init__(self, observation):
        obs = [DontCare() if str(o) == DontCare.symbol else o for o in
               observation]
        super().__init__(obs)

    def does_match(self, p: Perception) -> bool:
        for ci, oi in zip(self, p):
            if type(ci) is not DontCare and ci != oi:
                return False

        return True

    def subsumes(self, other) -> bool:
        raise NotImplementedError('MACS has no subsume operator')


class Effect(ImmutableSequence):
    WILDCARD = '?'  # don't know symbol - matches any value

    def does_match(self, p: Perception) -> bool:
        return all(ei == pi for ei, pi in zip(self, p) if ei != self.WILDCARD)

    def subsumes(self, other) -> bool:
        raise NotImplementedError('MACS has no subsume operator')


class Configuration:
    def __init__(self,
                 classifier_length: int,
                 number_of_possible_actions: int,
                 inaccuracy_threshold: int = 5,
                 accuracy_threshold: int = 5,
                 specified_symbols: int = 1):
        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.er = inaccuracy_threshold
        self.ea = accuracy_threshold
        self.specified_symbols = specified_symbols


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
        self.g = 0  # Number of good anticipations
        self.b = 0  # Number of bad anticipations

    @property
    def is_inaccurate(self) -> bool:
        return self.g == 0 and self.b == self.cfg.er

    def does_match(self, situation: Perception) -> bool:
        return self.condition.does_match(situation)

    def anticipates(self, situation: Perception) -> bool:
        return self.effect.does_match(situation)


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
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def evaluate_classifiers(population: ClassifiersList,
                             p0: Perception,
                             action: int,
                             p1: Perception):

        match_set = population.form_match_set(p0)
        action_set = match_set.form_action_set(action)

        for cl in action_set:
            if cl.anticipates(p1):
                cl.g += 1
            else:
                cl.b += 1

            if cl.is_inaccurate:
                population.safe_remove(cl)


class MACS(Agent):

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
        logging.debug("Running trial explore")

        # Initial conditions
        steps = 0
        last_reward = 0
        raw_state = env.reset()

        state = Perception(raw_state)
        prev_state = None

        action = None

        done = False

        while not done:
            logging.debug(f"Step {steps}, perception: {state}")

            # Select an action
            action = random.randint(0, self.cfg.number_of_possible_actions - 1)

            # Act in environment
            logging.debug(f"Executing action {action}")
            raw_state, last_reward, done, _ = env.step(action)

            if last_reward > 0:
                logging.debug("FOUND REWARD")

            prev_state = state
            state = Perception(raw_state)

            steps += 1

        return TrialMetrics(steps, last_reward)
