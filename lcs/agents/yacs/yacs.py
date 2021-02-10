from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum
from itertools import groupby
from typing import Union, Optional, Generator, List, Dict

from lcs import TypedList, Perception
from lcs.agents import ImmutableSequence, Agent
from lcs.agents.Agent import TrialMetrics

logging.basicConfig(level=logging.DEBUG)


@dataclass
class DontCare:
    symbol: str = '#'
    eis: float = 0.0  # expected improvement by specialization

    def __repr__(self):
        return self.symbol


class ClassifierTrace(Enum):
    BAD = 0
    GOOD = 1


class Configuration:
    def __init__(self,
                 classifier_length: int,
                 number_of_possible_actions: int,
                 feature_possible_values: list,
                 trace_length: int = 5,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9):
        assert classifier_length == len(feature_possible_values)
        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.feature_possible_values = feature_possible_values
        self.trace_length = trace_length
        self.beta = learning_rate
        self.gamma = discount_factor


class Condition(ImmutableSequence):
    WILDCARD = DontCare()
    OK_TYPES = (str, DontCare)

    def __init__(self, observation):
        obs = [DontCare() if str(o) == DontCare.symbol else o for o in
               observation]
        super().__init__(obs)

    @property
    def expected_improvements(self) -> List[float]:
        return [f.eis if type(f) is DontCare else 0.0 for f in self]

    def does_match(self, p: Perception) -> bool:
        for ci, oi in zip(self, p):
            if ci != self.WILDCARD and ci != oi:
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

    def passthrough(self, obs: Perception) -> Perception:
        # Predicts next state (reverse of diff)
        return Perception(
            [ei if ei != ImmutableSequence.WILDCARD else pi for ei, pi in
             zip(self, obs)])

    def subsumes(self, other) -> bool:
        raise NotImplementedError('YACS has no subsume operator')


class Classifier:
    def __init__(self,
                 condition: Union[Condition, str, None] = None,
                 action: Optional[int] = None,
                 effect: Union[Effect, str, None] = None,
                 reward: float = 0,  # immediate reward
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
        self.r = reward
        self.trace = deque(maxlen=self.cfg.trace_length)
        self.last_bad_state = None  # situation preceding wrong anticipation
        self.last_good_state = None  # situation preceding good anticipation

    def __repr__(self):
        return f"{self.condition}-{self.action}-{self.effect} @ {hex(id(self))}"

    @property
    def trace_full(self) -> bool:
        return len(self.trace) == self.cfg.trace_length

    @property
    def oscillating(self) -> bool:
        return all(t in self.trace for t in
                   [ClassifierTrace.GOOD, ClassifierTrace.BAD])

    def does_match(self, situation: Perception) -> bool:
        return self.condition.does_match(situation)

    def add_to_trace(self, mark: ClassifierTrace):
        self.trace.append(mark)

    def is_specializable(self) -> bool:
        return self.trace_full and self.oscillating

    def memorize_state(self, p: Perception):
        # check the last trace
        last_anticipation_result = self.trace[-1]
        assert last_anticipation_result is not None

        if last_anticipation_result == ClassifierTrace.GOOD:
            self.last_good_state = p
        else:
            self.last_bad_state = p

    def update_reward(self, env_reward):
        self.r = (1 - self.cfg.beta) * self.r + self.cfg.beta * env_reward


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

    def specialize(self, pop: ClassifiersList):
        def keyfunc(cl):
            return repr(cl.condition), repr(cl.action)

        for key, grouped in groupby(sorted(pop, key=keyfunc), key=keyfunc):
            cl_group = list(grouped)
            if all(cl.is_specializable() for cl in cl_group):
                # Generate new classifiers with mutspec operator
                # by specializing condition parts
                new_cls = self.specialize_condition(cl_group)

                # Add newly created classifiers to initial population
                pop.extend(new_cls)

                # Remove all classifiers from the population
                for cl in cl_group:
                    pop.remove(cl)

    @staticmethod
    def select_accurate_classifiers(population: ClassifiersList):
        for cl in population:
            if cl.trace_full and cl.oscillating:
                population.remove(cl)

    def specialize_condition(self, pop: Union[list, ClassifiersList]) -> Generator[Classifier]:
        assert len(pop) > 0
        eis = [cl.condition.expected_improvements for cl in pop]
        summed_eis = [sum(x) for x in zip(*eis)]

        feature_idx = summed_eis.index(max(summed_eis))

        yield from self.mutspec(pop[0], feature_idx)

    def mutspec(self, cl: Classifier, feature_idx: int) -> Generator[Classifier]:
        assert type(cl.condition[feature_idx]) == DontCare

        for feature in range(self.cfg.feature_possible_values[feature_idx]):
            # Build condition (TODO: eis are reseted here...)
            new_c = Condition(cl.condition)
            new_c[feature_idx] = str(feature)

            # Build effect
            new_e = Effect(cl.effect)
            if new_c[feature_idx] == new_e[feature_idx]:
                new_e[feature_idx] = DontCare.symbol

            yield Classifier(
                condition=new_c,
                action=cl.action,
                effect=new_e,
                cfg=cl.cfg
            )


class PolicyLearning:

    def __init__(self, cfg: Configuration):
        self.cfg = cfg

    @staticmethod
    def update_immediate_rewards(pop: ClassifiersList,
                                 obs: Perception,
                                 action: int,
                                 env_reward: int):

        match_set = pop.form_match_set(obs)
        action_set = match_set.form_action_set(action)
        for cl in action_set:
            cl.update_reward(env_reward)

    def update_desirability_values(self,
                                   pop: ClassifiersList,
                                   desirability_values: Dict[Perception, float],
                                   fired_cl: Classifier,
                                   obs: Perception):

        assert obs in desirability_values
        match_set = pop.form_match_set(obs)

        if fired_cl in match_set:
            max_r = max(cl.r for cl in match_set)
            anticipated_obs = fired_cl.effect.passthrough(obs)
            desirability_values[obs] = max_r + self.cfg.gamma * desirability_values.get(anticipated_obs, 0.0)

    def select_action(self,
                      pop: ClassifiersList,
                      desirability_values: Dict[Perception, float],
                      obs: Perception) -> int:

        match_set = pop.form_match_set(obs)

        def quality(cl: Classifier):
            anticipated_obs = cl.effect.passthrough(obs)
            return cl.r + self.cfg.gamma * desirability_values.get(anticipated_obs, 0.0)

        selected_cl = max(match_set, key=quality)
        return selected_cl.action


class YACS(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None,
                 desirability_values: Dict[Perception, float] = None):
        self.cfg = cfg
        self.desirability_values = desirability_values or dict()
        self.population = population or ClassifiersList()
        self.ll = LatentLearning(cfg)
        self.pl = PolicyLearning(cfg)

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def remember_situation(self, p: Perception):
        assert len(p) == self.cfg.classifier_length

        for f_max, p in zip(self.cfg.feature_possible_values, p):
            assert int(p) in range(0, f_max)

        if p not in self.desirability_values:
            self.desirability_values[p] = 0.0

    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        logging.info("Running trial explore")

        # Initial conditions
        steps = 0
        last_reward = 0
        raw_state = env.reset()

        state = Perception(raw_state)
        prev_state = Perception.empty()

        selected_cl = None
        prev_selected_cl = None

        done = False

        while not done:
            agent.remember_situation(state)
            match_set = self.population.form_match_set(state)

            if len(match_set) == 0:
                # TODO: cover new classifier, add to match-set and population
                pass

            if steps > 0:
                # Latent learning
                self.ll.effect_covering(
                    self.population,
                    prev_state,
                    state,
                    prev_selected_cl.action)
                self.ll.select_accurate_classifiers(self.population)
                self.ll.specialize(self.population)

                # Policy learning
                self.pl.update_immediate_rewards(
                    self.population,
                    prev_state,
                    prev_selected_cl.action,
                    last_reward)
                self.pl.update_desirability_values(
                    self.population,
                    self.desirability_values,
                    prev_selected_cl,
                    prev_state)

            # Select an action
            prev_selected_cl = selected_cl
            selected_cl = random.choice(match_set)
            action = selected_cl.action

            # Act in environment
            raw_state, last_reward, done, _ = env.step(action)
            prev_state = state
            state = Perception(raw_state)

            steps += 1

        return TrialMetrics(steps, last_reward)

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        logging.info("Running trial exploit")
        return None


if __name__ == '__main__':
    cfg = Configuration(4, 2, feature_possible_values=[2, 2, 2, 2])
    agent = YACS(cfg)
