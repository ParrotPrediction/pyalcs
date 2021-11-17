from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum
from itertools import groupby
from typing import Union, Optional, Generator, List, Dict, Callable

from lcs import TypedList, Perception
from lcs.agents import ImmutableSequence, Agent
from lcs.agents.Agent import TrialMetrics


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
    def __init__(self, **kwargs):
        # length of the classifier
        self.classifier_length: int = kwargs.get('classifier_length')

        # number of possible actions
        self.number_of_possible_actions: int = kwargs.get(
            'number_of_possible_actions')

        self.feature_possible_values: list = kwargs.get(
            'feature_possible_values')

        self.trace_length: int = kwargs.get('trace_length', 5)

        # learning rate
        self.beta: float = kwargs.get('beta', 0.1)

        # discount factor
        self.gamma: float = kwargs.get('gamma', 0.9)

        # whether to estimate the expected improvement of each attribute
        self.estimate_expected_improvements: bool = kwargs.get(
            'estimate_expected_improvements', True)

        # frequency of collecting metrics
        self.metrics_trial_frequency: int = kwargs.get(
            'metrics_trial_frequency', 1)

        # custom function for collecting customized metrics
        self.user_metrics_collector_fcn: Optional[Callable] = kwargs.get(
            'user_metrics_collector_fcn', None)

        # whether to use mlflow
        self.use_mlflow: bool = kwargs.get('use_mlflow', False)

        # how often dump model object with mlflow
        self.model_checkpoint_freq: Optional[int] = kwargs.get(
            'model_checkpoint_frequency', None)

        assert self.classifier_length == len(self.feature_possible_values)


class Condition(ImmutableSequence):
    WILDCARD = DontCare()
    OK_TYPES = (str, DontCare)

    def __init__(self, observation):
        obs = [DontCare() if str(o) == DontCare.symbol else o for o in
               observation]
        super().__init__(obs)

    @staticmethod
    def random_matching(p0: Perception) -> Generator[Condition]:
        max_val = len(p0) - 1
        while True:
            c = Condition(p0)
            num_dont_cares = random.randint(0, max_val)
            for i in random.sample(range(0, max_val), num_dont_cares):
                c[i] = DontCare()
            yield c

    @property
    def expected_improvements(self) -> List[float]:
        return [f.eis if type(f) is DontCare else None for f in self]

    @property
    def generality(self) -> int:
        return sum(1 for f in self if type(f) is DontCare)

    def is_more_specialized(self, other: Condition) -> bool:
        """Checks if other condition is more specialized"""

        def is_less_general(s, o):
            if type(s) is DontCare:
                return True

            return s == o

        def is_different(s, o):
            if type(s) is DontCare and type(o) is DontCare:
                return False
            else:
                return s != o

        other_less_general = all(
            is_less_general(s, o) for s, o in zip(self, other))
        different_tokens = sum(
            1 for s, o in zip(self, other) if is_different(s, o))

        return other_less_general and different_tokens > 0

    def is_more_general(self, other: Condition) -> bool:
        """Checks if other condition is more general"""

        def is_more_general(s, o):
            if type(o) is DontCare:
                return True

            return s == o

        def is_different(s, o):
            if type(s) is DontCare and type(o) is DontCare:
                return False
            else:
                return s != o

        other_more_general = all(
            is_more_general(s, o) for s, o in zip(self, other))
        different_tokens = sum(
            1 for s, o in zip(self, other) if is_different(s, o))

        return other_more_general and different_tokens > 0

    @property
    def specificity(self) -> int:
        return len(self) - self.generality

    def does_match(self, p: Perception) -> bool:
        for ci, oi in zip(self, p):
            if type(ci) is not DontCare and ci != oi:
                return False

        return True

    def increase_eis(self, idx, beta):
        t = self[idx]
        if type(t) is DontCare:
            t.eis = (1 - beta) * t.eis + beta

    def decrease_eis(self, idx, beta):
        t = self[idx]
        if type(t) is DontCare:
            t.eis = (1 - beta) * t.eis

    def subsumes(self, other) -> bool:
        raise NotImplementedError('YACS has no subsume operator')


class Effect(ImmutableSequence):
    @staticmethod
    def diff(p0: Optional[Perception], p1: Perception) -> Effect:
        # Computes the desired effect
        if p0 is None:
            return Effect(p1)

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
                 debug: dict = dict(),
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
        self.last_bad_perception: Optional[
            Perception] = None  # situation preceding wrong anticipation
        self.last_good_perception: Optional[
            Perception] = None  # situation preceding good anticipation
        self.debug = debug

    def __repr__(self):
        return f"{self.condition}-{self.action}-{self.effect} @ {hex(id(self))}"

    @property
    def trace_full(self) -> bool:
        return len(self.trace) == self.cfg.trace_length

    @property
    def oscillating(self) -> bool:
        return all(t in self.trace for t in
                   [ClassifierTrace.GOOD, ClassifierTrace.BAD])

    def is_reliable(self) -> bool:
        return not self.oscillating

    def anticipation(self, obs: Perception) -> Perception:
        return self.effect.passthrough(obs)

    def does_match(self, situation: Perception) -> bool:
        return self.condition.does_match(situation)

    def add_to_trace(self, mark: ClassifierTrace):
        self.trace.append(mark)

    def is_specializable(self) -> bool:
        return self.trace_full and self.oscillating

    def update_reward(self, env_reward):
        self.r = (1 - self.cfg.beta) * self.r + self.cfg.beta * env_reward

    def predicts_successfully(self,
                              p0: Perception,
                              action: int,
                              p1: Perception) -> bool:
        if self.does_match(p0):
            if self.action == action:
                if self.effect == Effect.diff(p0, p1):
                    return True

        return False


class ClassifiersList(TypedList[Classifier]):
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

    def cover_classifier(self,
                         population: ClassifiersList,
                         action: int,
                         p0: Optional[Perception],
                         p1: Perception) -> Optional[Classifier]:

        def _neither_more_general_nor_more_specialized(
            cond: Condition, pop: ClassifiersList) -> bool:

            more_general_exists = any(
                True for cl in pop if cond.is_more_general(cl.condition))
            more_specialized_exists = any(
                True for cl in pop if cond.is_more_specialized(cl.condition))

            return more_general_exists is False and more_specialized_exists is False

        action_set = population.form_action_set(action)

        # generate condition until desired conditions are met
        c = None
        if len(action_set) == 0:
            c = [DontCare()] * self.cfg.classifier_length
        else:
            tries = 1000
            while tries > 0:
                c = next(Condition.random_matching(p1))
                if _neither_more_general_nor_more_specialized(c, action_set):
                    break

                tries -= 1
                if tries == 0:
                    logging.warning(
                        f'Unable to cover classifier for perception: {p1}')
                    return None

        # Create effect part
        e = Effect.diff(p0, p1)
        for idx, (ci, ei) in enumerate(zip(c, e)):
            if ci == ei:
                e[idx] = ImmutableSequence.WILDCARD

        return Classifier(
            condition=Condition(c),
            action=action,
            effect=e,
            debug={'origin': 'covering'},
            cfg=self.cfg)

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
            cl.last_good_perception = p0

        # Add trace to classifiers that anticipated wrong
        wrong_classifiers = [cl for cl in action_set if cl.effect != de]
        for cl in wrong_classifiers:
            cl.add_to_trace(ClassifierTrace.BAD)
            cl.last_bad_perception = p0

        if self.cfg.estimate_expected_improvements:
            # Adjust expected improvement by specialization values
            for cl in [cl for cl in good_classifiers if
                       cl.last_bad_perception is not None]:
                for i, (p0i, bpi) in enumerate(
                    zip(p0, cl.last_bad_perception)):
                    if p0i == bpi:
                        cl.condition.decrease_eis(i, self.cfg.beta)
                    else:
                        cl.condition.increase_eis(i, self.cfg.beta)

            for cl in [cl for cl in wrong_classifiers if
                       cl.last_good_perception is not None]:
                for i, (p0i, gpi) in enumerate(
                    zip(p0, cl.last_good_perception)):
                    if p0i == gpi:
                        cl.condition.decrease_eis(i, self.cfg.beta)
                    else:
                        cl.condition.increase_eis(i, self.cfg.beta)

        # If no classifier has correct anticipation - create it
        if len(good_classifiers) == 0 and len(wrong_classifiers) > 0:
            old_cl = random.choice(wrong_classifiers)
            new_cl = Classifier(
                condition=Condition(old_cl.condition),
                action=old_cl.action,
                effect=de,
                debug={'origin': 'effect_covering'},
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
            if cl.trace_full and not all(
                t == ClassifierTrace.GOOD for t in cl.trace):
                population.remove(cl)

    def specialize_condition(self, pop: Union[list, ClassifiersList]) -> \
        Generator[Classifier]:
        assert len(pop) > 0
        eis = [cl.condition.expected_improvements for cl in pop]

        summed_eis = [None] * self.cfg.classifier_length
        for idx, sei in enumerate(summed_eis):
            if all(x[idx] is not None for x in eis):
                summed_eis[idx] = sum(x[idx] for x in eis)

        if not all(val is None for val in summed_eis):
            if self.cfg.estimate_expected_improvements:
                feature_idx = max(
                    ((idx, val) for idx, val in enumerate(summed_eis) if
                     val is not None), key=lambda x: x[1])[0]
            else:
                feature_idx = random.choice(
                    [idx for idx, val in enumerate(summed_eis) if val is not None])

            for cl in pop:
                yield from self.mutspec(cl, feature_idx)

    def mutspec(self, cl: Classifier, feature_idx: int) -> Generator[
        Classifier]:
        assert type(cl.condition[feature_idx]) == DontCare

        for feature in self.cfg.feature_possible_values[feature_idx]:
            # Build condition
            new_c = Condition(cl.condition)
            new_c[feature_idx] = str(feature)

            # Build effect
            new_e = Effect(cl.effect)
            for idx, (ci, ei) in enumerate(zip(new_c, new_e)):
                if ci == ei:
                    new_e[idx] = ImmutableSequence.WILDCARD

            yield Classifier(
                condition=new_c,
                action=cl.action,
                effect=new_e,
                debug={'origin': 'mutespec'},
                cfg=cl.cfg
            )


class PolicyLearning:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg

    def update_optimal_policy(self,
                              pop: ClassifiersList,
                              desirability_values: Dict[Perception, float],
                              obs: Perception,
                              action: int,
                              env_reward: int):
        assert obs in desirability_values
        match_set = pop.form_match_set(obs)
        action_set = match_set.form_action_set(action)

        # Assign immediate rewards
        for cl in action_set:
            cl.update_reward(env_reward)

        # Update desirability value
        anticipated_obs = [cl.anticipation(obs) for cl in action_set]
        exp_cum_reward = max((v for p, v in desirability_values.items() if
                              p in anticipated_obs), default=0)

        desirability_values[obs] = env_reward + self.cfg.gamma * exp_cum_reward

    def select_action(self,
                      match_set: ClassifiersList,
                      desirability_values: Dict[Perception, float],
                      obs: Perception) -> int:

        def quality(cl: Classifier):
            anticipated_obs = cl.anticipation(obs)
            return cl.r + self.cfg.gamma * desirability_values.get(
                anticipated_obs, 0.0)

        if len(match_set) == 0:
            return random.randint(0, self.cfg.number_of_possible_actions - 1)

        selected_cl = max(match_set, key=quality)
        assert selected_cl.does_match(obs)

        return selected_cl.action


class YACS(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None,
                 desirability_values: Dict[Perception, float] = None):
        self.cfg = cfg
        self.population = population or ClassifiersList()
        self.desirability_values = desirability_values or dict()
        self.ll = LatentLearning(cfg)
        self.pl = PolicyLearning(cfg)

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def remember_situation(self, p: Perception):
        assert len(p) == self.cfg.classifier_length,\
            f'Perception [{p}]'

        for allowed, _p in zip(self.cfg.feature_possible_values, p):
            assert _p in allowed, f'value [{_p}] not allowed'

        if p not in self.desirability_values:
            self.desirability_values[p] = 0.0

    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        logging.debug("Running trial explore")

        # Initial conditions
        steps = 0
        last_reward = 0
        raw_state = env.reset()
        state = Perception(raw_state)

        done = False

        while not done:
            logging.debug(f"Step {steps}, perception: {state}")
            self.remember_situation(state)

            # Select an action
            action = random.randint(0, self.cfg.number_of_possible_actions - 1)

            # Act in environment
            logging.debug(f"Executing action {action}")
            raw_state, last_reward, done, _ = env.step(action)

            if last_reward > 0:
                logging.debug("FOUND REWARD")

            prev_state = state
            state = Perception(raw_state)

            match_set = self.population.form_match_set(prev_state)
            action_set = match_set.form_action_set(action)

            logging.debug(f"Action set size {len(action_set)}")

            if len(action_set) == 0:
                cl = self.ll.cover_classifier(self.population, action,
                                              prev_state, state)
                if cl is not None:
                    self.population.append(cl)

            self.ll.effect_covering(self.population, prev_state, state, action)
            self.ll.specialize(self.population)
            self.ll.select_accurate_classifiers(self.population)

            self.pl.update_optimal_policy(self.population,
                                          self.desirability_values, prev_state,
                                          action, last_reward)

            steps += 1

        return TrialMetrics(steps, last_reward)

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        logging.debug("Running trial exploit")

        # Initial conditions
        steps = 0
        last_reward = 0
        raw_state = env.reset()
        state = Perception(raw_state)
        done = False

        while not done:
            logging.info(f"\tStep {steps}, perception: {state}")

            # Select an action
            match_set = self.population.form_match_set(state)
            if len(match_set) == 0:
                logging.error(f"Unknown state [{state}] encountered. Please retrain the agent")

            action = self.pl.select_action(match_set, self.desirability_values, state)

            # Act in environment
            logging.info(f"\tExecuting action {action}")
            raw_state, last_reward, done, _ = env.step(action)

            if last_reward > 0:
                logging.debug("FOUND REWARD")

            state = Perception(raw_state)
            steps += 1

        return TrialMetrics(steps, last_reward)


if __name__ == '__main__':
    cfg = Configuration(
        classifier_length=4,
        number_of_possible_actions=2,
        feature_possible_values=[{0, 1}, {0, 1}, {0, 1}, {0, 1}])

    agent = YACS(cfg)
