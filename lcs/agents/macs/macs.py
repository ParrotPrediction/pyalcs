from __future__ import annotations

import logging
import random
from typing import Union, Optional, Dict, Generator, Set

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

    def does_match(self, p: Perception) -> bool:
        for ci, oi in zip(self, p):
            if ci != self.WILDCARD and ci != oi:
                return False

        return True

    def increase_eis(self, idx, beta):
        if self[idx] == self.WILDCARD:
            self.eis[idx] = (1 - beta) * self.eis[idx] + beta
        else:
            raise ValueError('Trying to modify eis for non wildcard')

    def decrease_eis(self, idx, beta):
        if self[idx] == self.WILDCARD:
            self.eis[idx] = (1 - beta) * self.eis[idx]
        else:
            raise ValueError('Trying to modify eis for non wildcard')

    def feature_to_specialize(self) -> Optional[int]:
        """Returns index of the feature suggested for specialization"""
        if all(c != self.WILDCARD for c in self):
            return None

        eis = {idx: self.eis[idx] for idx, c in enumerate(self) if c == self.WILDCARD}
        return max(eis, key=eis.get)

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
                 feature_possible_values: list,
                 learning_rate: float = 0.1,
                 inaccuracy_threshold: int = 5,
                 accuracy_threshold: int = 5,
                 oscillation_threshold: int = 5,
                 specified_symbols: int = 1):
        assert classifier_length == len(feature_possible_values)
        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.feature_possible_values = feature_possible_values
        self.beta = learning_rate
        self.er = inaccuracy_threshold
        self.ea = accuracy_threshold
        self.eo = oscillation_threshold
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

        # Number of good anticipations
        self.g = 0

        # Situation preceding last anticipation success
        self.sg: Optional[Perception] = None

        # Number of bad anticipations
        self.b = 0

        # Situation preceding last anticipation mistake
        self.sb: Optional[Perception] = None

    @property
    def is_inaccurate(self) -> bool:
        return self.g == 0 and self.b == self.cfg.er

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
                if cl.sb is not None:
                    for i, (p0i, bpi) in enumerate(zip(p0, cl.sb)):
                        if p0i == bpi:
                            cl.condition.decrease_eis(i, self.cfg.beta)
                        else:
                            cl.condition.increase_eis(i, self.cfg.beta)
            else:
                cl.b += 1
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
                              perceptions: Set[Perception]) -> None:

        for cl in [cl for cl in pop if cl.is_oscillating]:
            feature_idx = cl.condition.feature_to_specialize()
            for new_cl in self.mutspec(cl, feature_idx):
                if any(new_cl.does_match(p) for p in perceptions):
                    pop.append(new_cl)

    def mutspec(self, cl: Classifier, feature_idx: int) -> Generator[Classifier]:
        assert cl.condition[feature_idx] == Condition.WILDCARD
        for feature in range(self.cfg.feature_possible_values[feature_idx]):
            new_c = Condition(cl.condition)
            new_c[feature_idx] = str(feature)

            yield Classifier(
                condition=new_c,
                action=cl.action,
                effect=Effect(cl.effect),
                cfg=cl.cfg
            )


class MACS(Agent):
    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None,
                 desirability_values: Dict[Perception, float] = None):
        self.cfg = cfg
        self.population = population or ClassifiersList()
        self.desirability_values = desirability_values or dict()

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def remember_situation(self, p: Perception):
        assert len(p) == self.cfg.classifier_length

        for f_max, _p in zip(self.cfg.feature_possible_values, p):
            assert int(_p) in range(0, f_max)

        if p not in self.desirability_values:
            self.desirability_values[p] = 0.0

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


if __name__ == '__main__':
    cfg = Configuration(4, 2, feature_possible_values=[2, 2, 2, 2])
    agent = MACS(cfg)
