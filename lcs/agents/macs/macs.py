from __future__ import annotations

import itertools
import logging
import random
from itertools import groupby
from operator import attrgetter
from typing import Union, Optional, Dict, Generator, Set, Tuple, NamedTuple, \
    Callable

from lcs import TypedList, Perception
from lcs.agents import Agent, ImmutableSequence
from lcs.agents.Agent import TrialMetrics


class Condition(ImmutableSequence):
    def __init__(self, observation):
        super().__init__(observation)

        # expected improvement by specialization
        self.eis = [0.5] * len(observation)

        # improvements by generalization
        self.ig = [0.5] * len(observation)

    def __lt__(self, other: Condition):
        return other.generality < self.generality

    @classmethod
    def general(cls, length: int) -> Condition:
        return Condition([cls.WILDCARD] * length)

    @property
    def generality(self) -> int:
        return sum(1 for c in self if c == self.WILDCARD)

    def does_match(self, p: Perception) -> bool:
        return all(ci == pi for ci, pi in zip(self, p) if ci != self.WILDCARD)

    def non_matching(self, o: Condition) -> bool:
        """
        Checks if other condition is covered by current one.
        """
        if self == o:
            return True

        return any(ci != oi for ci, oi in zip(self, o) if ci != self.WILDCARD)

    @staticmethod
    def generate_matching(p: Perception) -> Generator[Condition]:
        wildcards = len(p)
        yield Condition(p)

        while wildcards > 0:
            combinations = itertools.combinations(range(0, len(p)), wildcards)

            for combination in combinations:
                c = Condition(p)
                for c_id in combination:
                    c[c_id] = Condition.WILDCARD
                yield c

            wildcards -= 1

    def increase_eis(self, idx, beta):
        if self[idx] == self.WILDCARD:
            self.eis[idx] = (1 - beta) * self.eis[idx] + beta

    def decrease_eis(self, idx, beta):
        if self[idx] == self.WILDCARD:
            self.eis[idx] = (1 - beta) * self.eis[idx]

    def increase_ig(self, idx, beta):
        if self[idx] != self.WILDCARD:
            self.ig[idx] = (1 - beta) * self.ig[idx] + beta
        else:
            raise ValueError('Trying to modify ig for a wildcard')

    def decrease_ig(self, idx, beta):
        if self[idx] != self.WILDCARD:
            self.ig[idx] = (1 - beta) * self.ig[idx]
        else:
            raise ValueError('Trying to modify if for a wildcard')

    def is_more_general(self, other: Condition) -> bool:
        """Checks if other condition is more general"""
        if self == other:
            return True

        # TODO validate if method below are useful
        def is_more_general(s, o):
            if o == self.WILDCARD:
                return True

            return s == o

        def is_different(s, o):
            if s == self.WILDCARD and o == self.WILDCARD:
                return False
            else:
                return s != o

        other_more_general = all(
            is_more_general(s, o) for s, o in zip(self, other))
        different_tokens = sum(
            1 for s, o in zip(self, other) if is_different(s, o))

        return other_more_general and different_tokens > 0

    def feature_to_specialize(self, estimate_expected_improvements: bool) -> Optional[int]:
        """Returns index of the feature suggested for specialization"""
        if all(c != self.WILDCARD for c in self):
            return None

        eis = {idx: self.eis[idx] for idx, c in enumerate(self) if
               c == self.WILDCARD}

        if estimate_expected_improvements:
            return max(eis, key=eis.get)
        else:
            return random.choice(list(eis.keys()))

    def feature_to_generalize(self) -> Optional[int]:
        """Returns index of the feature suggested for generalization"""
        if all(c == self.WILDCARD for c in self):
            return None

        igs = {idx: self.ig[idx] for idx, c in enumerate(self) if
               c != self.WILDCARD}
        return max(igs, key=igs.get)

    def exhaustive_generalization(self) -> Generator[Tuple[Condition, int]]:
        """
        Generates new condition where each specialized attribute is generalized
        """
        spec_idx = [idx for idx, c in enumerate(self) if c != self.WILDCARD]
        for spec_id in spec_idx:
            new_obs = list(self._items[:])
            new_obs[spec_id] = self.WILDCARD
            yield Condition(new_obs), spec_id

    def subsumes(self, other) -> bool:
        raise NotImplementedError('MACS has no subsume operator')


class Effect(ImmutableSequence):
    WILDCARD = '?'  # don't know symbol - matches any value

    @staticmethod
    def generate(p: Perception, specific_count: int = 1) -> Generator[Effect]:
        str_len = len(p)
        combinations = itertools.combinations(
            range(0, str_len), str_len - specific_count)

        for combination in combinations:
            items = list(p)

            for wildcard_id in combination:
                items[wildcard_id] = Effect.WILDCARD

            yield Effect(items)

    def __lt__(self, other: Effect):
        return self._items < other._items

    def conflicts(self, other: Effect) -> bool:
        """Other classifier has different value for the same attribute"""
        for si, oi in zip(self, other):
            if si != self.WILDCARD and oi != self.WILDCARD and si != oi:
                return True

        return False

    def does_match(self, p: Perception) -> bool:
        return all(ei == pi for ei, pi in zip(self, p) if ei != self.WILDCARD)

    def subsumes(self, other) -> bool:
        raise NotImplementedError('MACS has no subsume operator')


class Configuration:
    def __init__(self,
                 classifier_length: int,
                 number_of_possible_actions: int,
                 feature_possible_values: list,
                 specified_effect_attributes: int = 1,
                 estimate_expected_improvements: bool = True,
                 learning_rate: float = 0.1,
                 inaccuracy_threshold: int = 5,
                 accuracy_threshold: int = 5,
                 oscillation_threshold: int = 5,
                 metrics_trial_frequency: int = 5,
                 user_metrics_collector_fcn: Callable = None):
        assert classifier_length == len(feature_possible_values)
        assert 1 <= specified_effect_attributes <= classifier_length

        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.feature_possible_values = feature_possible_values
        self.specified_effect_attributes = specified_effect_attributes
        self.estimate_expected_improvements = estimate_expected_improvements
        self.beta = learning_rate
        self.er = inaccuracy_threshold
        self.ea = accuracy_threshold
        self.eo = oscillation_threshold
        self.metrics_trial_frequency = metrics_trial_frequency
        self.user_metrics_collector_fcn = user_metrics_collector_fcn


class Classifier:
    def __init__(self,
                 condition: Union[Condition, str, None] = None,
                 action: Optional[int] = None,
                 effect: Union[Effect, str, None] = None,
                 debug: dict = dict(),
                 cfg: Optional[Configuration] = None):

        if cfg is None:
            raise TypeError("Configuration should be passed to Classifier")

        self.cfg = cfg
        self.debug = debug

        def build_perception_string(cls, initial,
                                    length=self.cfg.classifier_length):
            if initial:
                return cls(initial)

            return cls.empty(length=length)

        self.condition = build_perception_string(Condition, condition)
        self.action = action
        self.effect = build_perception_string(Effect, effect)

        # Number of good anticipations
        self.g = 0

        # Situation preceding last anticipation success
        self.sg: Optional[Perception] = None

        # Number of bad anticipations
        self.b = 0

        # Situation preceding last anticipation mistake
        self.sb: Optional[Perception] = None

    def __repr__(self):
        return f"{self.condition}-{self.action}-{self.effect}, G: {self.g}, B: {self.b}, {self.debug}"

    @property
    def is_accurate(self) -> bool:
        return self.b == 0 and self.g >= self.cfg.ea

    @property
    def is_inaccurate(self) -> bool:
        return self.g == 0 and self.b >= self.cfg.er

    @property
    def is_oscillating(self) -> bool:
        return self.g + self.b > self.cfg.eo and self.g * self.b > 0

    def does_match(self, situation: Perception) -> bool:
        return self.condition.does_match(situation)

    def anticipates(self, situation: Perception) -> bool:
        return self.effect.does_match(situation)


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
    def __init__(self, cfg: Configuration):
        self.cfg = cfg

    def evaluate_classifiers(self,
                             population: ClassifiersList,
                             p0: Perception,
                             action: int,
                             p1: Perception):

        match_set = population.form_match_set(p0)
        action_set = match_set.form_action_set(action)

        for cl in action_set:
            if cl.anticipates(p1):
                cl.g += 1
                cl.sg = p0
                if cl.sb is not None:
                    for i, (p0i, bpi) in enumerate(zip(p0, cl.sb)):
                        if p0i == bpi:
                            cl.condition.decrease_eis(i, self.cfg.beta)
                        else:
                            cl.condition.increase_eis(i, self.cfg.beta)
            else:
                cl.b += 1
                cl.sb = p0
                if cl.sg is not None:
                    for i, (p0i, gpi) in enumerate(zip(p0, cl.sg)):
                        if p0i == gpi:
                            cl.condition.decrease_eis(i, self.cfg.beta)
                        else:
                            cl.condition.increase_eis(i, self.cfg.beta)

    def select_accurate(self, pop: ClassifiersList) -> None:
        for cl in pop:
            if cl.is_inaccurate:
                pop.safe_remove(cl)

    def specialize_conditions(self,
                              pop: ClassifiersList,
                              situations_seen: Set[Perception]) -> None:

        for cl in [cl for cl in pop if cl.is_oscillating]:
            feature_idx = cl.condition.feature_to_specialize(
                self.cfg.estimate_expected_improvements)

            for new_cl in self.mutspec(cl, feature_idx):
                if any(new_cl.does_match(p) for p in situations_seen):
                    assert new_cl not in pop
                    pop.append(new_cl)

            pop.safe_remove(cl)

    def mutspec(self, cl: Classifier, feature_idx: int) -> Generator[
        Classifier]:
        assert cl.condition[feature_idx] == Condition.WILDCARD
        for feature in self.cfg.feature_possible_values[feature_idx]:
            new_c = Condition(cl.condition)
            new_c[feature_idx] = str(feature)

            yield Classifier(
                condition=new_c,
                action=cl.action,
                effect=Effect(cl.effect),
                debug={'origin': 'mutspec'},
                cfg=cl.cfg
            )

    def generalize_conditions(self,
                              population: ClassifiersList,
                              situations_seen: Set[Perception],
                              p0: Perception,
                              a0: int,
                              p1: Perception) -> None:

        set_a = self._update_igs(population, p0, a0, p1)

        set_c: Set[ChildClassifier] = set()  # general set of classifiers
        ChildClassifier = NamedTuple('ChildClassifier',
                                     [('classifier', Classifier),
                                      ('parent', Optional[Classifier])])

        # Sort classifiers by effect for grouping
        set_a = sorted(set_a, key=attrgetter("effect"))
        set_d: Set[Classifier] = set()  # conflicts

        for _, set_b in groupby(set_a, key=attrgetter("effect")):
            set_b = set(set_b)
            if all(cl.is_accurate for cl in set_b):
                # Build [C] only when all classifiers from [B] are accurate
                for cl in set_b:
                    if all(ig <= 0.5 for ig in cl.condition.ig):
                        set_c.add(
                            ChildClassifier(cl, cl))
                    else:
                        spec_cond_idx = cl.condition.feature_to_generalize()
                        assert spec_cond_idx is not None

                        new_cond = Condition(cl.condition)
                        new_cond[spec_cond_idx] = Condition.WILDCARD
                        new_cl = Classifier(condition=new_cond,
                                            action=cl.action,
                                            effect=Effect(cl.effect),
                                            debug={'origin': 'set_c'},
                                            cfg=cl.cfg)

                        set_c.add(ChildClassifier(new_cl, cl))

            # Check for conflicts in [C]
            for p in situations_seen:
                existing_cls = [cl for cl in population if
                                cl.does_match(p) and cl.action == a0]
                new_cls = [cl for cl in set_c if cl.classifier.does_match(
                    p) and cl.classifier.action == a0]

                for (existing_cl, new_cl) in itertools.product(
                    *[existing_cls, new_cls]):
                    if existing_cl.effect.conflicts(new_cl.classifier.effect):
                        assert new_cl.parent is not None
                        set_d.add(new_cl.parent)
                    else:
                        set_d.add(new_cl.classifier)

            # Analyze every possible pair of [D] for duplicates
            # (keep most general)
            for (c1, c2) in itertools.combinations(set_d, 2):
                if c1.condition.is_more_general(c2.condition) and c1 in set_d:
                    set_d.remove(c1)
                elif c2.condition.is_more_general(
                    c1.condition) and c2 in set_d:
                    set_d.remove(c2)

        # In the original population replace classifiers from [A]
        # by those generated from [D]
        for cl in set_a:
            population.remove(cl)

        for cl in set_d:
            assert cl not in population
            population.append(cl)

    def cover_transitions(self,
                          population: ClassifiersList,
                          p0: Perception,
                          a0: int,
                          p1: Perception) -> None:
        for ef in Effect.generate(p1, self.cfg.specified_effect_attributes):

            if len([cl.condition for cl in population if cl.does_match(p0) and cl.action == a0 and cl.effect == ef]) == 0:

                conditions = [cl.condition for cl in population if cl.action == a0 and cl.effect == ef]

                def non_matching_existing(generated: Condition) -> bool:
                    return all(generated.non_matching(c) for c in conditions)

                possible_conditions = list(filter(non_matching_existing, Condition.generate_matching(p0)))

                foo = dict()
                for k, v in itertools.groupby(possible_conditions, key=lambda c: c.generality):
                    foo[k] = list(v)

                c = None
                max_generalization = max(foo, key=int)
                for gcount in reversed(range(max_generalization + 1)):
                    conditions = foo[gcount]
                    if len(conditions):
                        c = random.choice(conditions)
                        break

                new_cl = Classifier(
                    condition=c,
                    action=a0,
                    effect=ef,
                    debug={'origin': 'covering'},
                    cfg=self.cfg)

                assert new_cl not in population
                population.append(new_cl)

    def _update_igs(self,
                    population: ClassifiersList,
                    p0: Perception,
                    a0: int,
                    p1: Perception) -> Set[Classifier]:
        """
        Compute estimates ig for each classifier.
        Returns classifiers whose A part patches a0 and whose E part matches p1
        """
        set_a: Set[Classifier] = set()
        non_matching = [cl for cl in population if
                        not cl.does_match(p0) and cl.action == a0]

        for cl in non_matching:
            for new_cond, idx in cl.condition.exhaustive_generalization():
                if new_cond.does_match(p0):
                    if cl.effect.does_match(p1):
                        cl.condition.increase_ig(idx, self.cfg.beta)
                        set_a.add(cl)
                    else:
                        cl.condition.decrease_ig(idx, self.cfg.beta)

        return set_a


class MACS(Agent):
    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None,
                 desirability_values: Dict[Perception, float] = None):
        self.cfg = cfg
        self.population = population or ClassifiersList()
        self.desirability_values = desirability_values or dict()
        self.ll = LatentLearning(cfg)

    def get_population(self) -> ClassifiersList:
        return self.population

    def get_cfg(self) -> Configuration:
        return self.cfg

    def remember_situation(self, p: Perception):
        assert len(p) == self.cfg.classifier_length

        for f_vals, _p in zip(self.cfg.feature_possible_values, p):
            assert _p in f_vals

        if p not in self.desirability_values:
            self.desirability_values[p] = 0.0

    def get_anticipations(self, p0: Perception, a: int) -> Generator[
        Perception]:
        match_set = self.population.form_match_set(p0)
        action_set = match_set.form_action_set(a)

        effects = [cl.effect for cl in action_set]

        anticipated_attribs = [set() for _ in range(len(p0))]
        for pi in range(len(p0)):
            for e in effects:
                if e[pi] != Effect.WILDCARD:
                    anticipated_attribs[pi].update(e[pi])

        if all(len(aa) > 0 for aa in anticipated_attribs):
            yield from map(Perception, itertools.product(*anticipated_attribs))

    def assert_no_duplicates(self):
        pop = self.get_population()
        assert len(pop) == len(set(pop._items)), 'duplicate classifiers found'

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
            self.assert_no_duplicates()
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

            seen_situations = set(self.desirability_values.keys())

            self.ll.evaluate_classifiers(
                self.population, prev_state, action, state)
            self.ll.select_accurate(self.population)
            self.ll.specialize_conditions(self.population, seen_situations)
            self.ll.generalize_conditions(
                self.population, seen_situations, prev_state, action, state)
            self.ll.cover_transitions(
                self.population, prev_state, action, state)

            steps += 1

        return TrialMetrics(steps, last_reward)


if __name__ == '__main__':
    state_values = {0, 1}
    cfg = Configuration(4, 2, feature_possible_values=[state_values] * 4)
    agent = MACS(cfg)
