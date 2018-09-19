from __future__ import annotations

from itertools import chain
from random import random, sample
from typing import Optional, List

import lcs.agents.acs2.components.alp as alp
import lcs.strategies.genetic_algorithms as ga
from lcs import Perception, TypedList
from . import Classifier, Configuration


class ClassifiersList(TypedList):
    """
    Represents overall population, match/action sets
    """
    def __init__(self, *args, cfg: Configuration) -> None:
        self.cfg = cfg
        super().__init__((Classifier, ), *args)

    def form_match_set(self,
                       situation: Perception,
                       cfg: Configuration) -> ClassifiersList:
        matching = [cl for cl in self if cl.condition.does_match(situation)]
        return ClassifiersList(*matching, cfg=cfg)

    def form_action_set(self,
                        action: int,
                        cfg: Configuration) -> ClassifiersList:
        matching = [cl for cl in self if cl.action == action]
        return ClassifiersList(*matching, cfg=cfg)

    def expand(self) -> List[Classifier]:
        """
        Returns an array containing all micro-classifiers

        Returns
        -------
        List[Classifier]
            list of all expanded classifiers
        """
        list2d = [[cl] * cl.num for cl in self]
        return list(chain.from_iterable(list2d))

    def get_maximum_fitness(self) -> float:
        """
        Returns the maximum fitness value amongst those classifiers
        that anticipated a change in environment.

        Returns
        -------
        float
            fitness value
        """
        anticipated_change_cls = [cl for cl in self
                                  if cl.does_anticipate_change()]

        if len(anticipated_change_cls) > 0:
            best_cl = max(anticipated_change_cls, key=lambda cl: cl.fitness)
            return best_cl.fitness

        return 0.0

    def apply_alp(self,
                  p0: Perception,
                  action: int,
                  p1: Perception,
                  time: int,
                  population: ClassifiersList,
                  match_set: ClassifiersList) -> None:
        """
        The Anticipatory Learning Process. Handles all updates by the ALP,
        insertion of new classifiers in pop and possibly matchSet, and
        deletion of inadequate classifiers in pop and possibly matchSet.

        :param p0:
        :param action:
        :param p1:
        :param time:
        :param population:
        :param match_set:
        """
        new_list = ClassifiersList(cfg=self.cfg)
        new_cl: Optional[Classifier] = None
        was_expected_case = False
        delete_count = 0

        for cl in self:
            cl.increase_experience()
            cl.set_alp_timestamp(time)

            if cl.does_anticipate_correctly(p0, p1):
                new_cl = alp.expected_case(cl, p0, time)
                was_expected_case = True
            else:
                new_cl = alp.unexpected_case(cl, p0, p1, time)

                if cl.is_inadequate():
                    # Removes classifier from population, match set
                    # and current list
                    delete_count += 1
                    lists = [x for x in [population, match_set, self] if x]
                    for lst in lists:
                        lst.safe_remove(cl)

            if new_cl is not None:
                new_cl.tga = time
                self.add_alp_classifier(new_cl, new_list)

        # No classifier anticipated correctly - generate new one
        if not was_expected_case:
            new_cl = alp.cover(p0, action, p1, time, self.cfg)
            self.add_alp_classifier(new_cl, new_list)

        # Merge classifiers from new_list into self and population
        self.extend(new_list)
        population.extend(new_list)

        if match_set is not None:
            new_matching = [cl for cl in new_list if
                            cl.condition.does_match(p1)]
            match_set.extend(new_matching)

    def apply_reinforcement_learning(self, reward: int, p) -> None:
        """
        Reinforcement Learning. Applies RL according to
        current reinforcement `reward` and back-propagated reinforcement
        `maximum_fitness`.

        :param reward: current reward
        :param p: maximum fitness - back-propagated reinforcement
        """
        for cl in self:
            cl.update_reward(reward + self.cfg.gamma * p)
            cl.update_intermediate_reward(reward)

    @staticmethod
    def apply_ga(time: int,
                 population: ClassifiersList,
                 match_set: ClassifiersList,
                 action_set: ClassifiersList,
                 p: Perception,
                 theta_ga: int,
                 chi: float,
                 theta_as: int,
                 do_subsumption: bool,
                 randomfunc=random,
                 samplefunc=sample) -> None:

        if ga.should_apply(action_set, time, theta_ga):
            ga.set_timestamps(action_set, time)

            # Select parents
            parent1, parent2 = ga.roulette_wheel_selection(
                action_set, lambda cl: pow(cl.q, 3) * cl.num)

            child1 = Classifier.copy_from(parent1, time)
            child2 = Classifier.copy_from(parent2, time)

            ga.generalizing_mutation(child1, child1.cfg.mu)
            ga.generalizing_mutation(child2, child2.cfg.mu)

            if randomfunc() < chi:
                if child1.effect == child2.effect:
                    ga.two_point_crossover(child1, child2, samplefunc=samplefunc)

                    # Update quality and reward
                    child1.q = child2.q = float(sum([child1.q, child2.q]) / 2)
                    child2.r = child2.r = float(sum([child1.r, child2.r]) / 2)

            child1.q /= 2
            child2.q /= 2

            # we are interested only in classifiers with specialized condition
            unique_children = {cl for cl in [child1, child2]
                               if cl.condition.specificity > 0}

            ga.delete_classifiers(
                population, match_set, action_set,
                len(unique_children), theta_as)

            # check for subsumers / similar classifiers
            for child in unique_children:
                ga.add_classifier(child, p,
                                  population, match_set, action_set,
                                  do_subsumption)

    def add_alp_classifier(self,
                           child: Classifier,
                           new_list: ClassifiersList) -> None:
        """
        Looks for subsuming / similar classifiers in the current set and
        those created in the current ALP run.

        If a similar classifier was found it's quality is increased,
        otherwise `child_cl` is added to `new_list`.

        Parameters
        ----------
        child:  Classifier
            New classifier to examine
        new_list: ClassifiersList
            A list of newly created classifiers in this ALP run
        """
        # TODO: p0: write tests
        old_cl = None

        # Look if there is a classifier that subsumes the insertion candidate
        for cl in self:
            if cl.does_subsume(child):
                if old_cl is None or cl.is_more_general(old_cl):
                    old_cl = cl

        # Check if any similar classifier was in this ALP run
        if old_cl is None:
            for cl in new_list:
                if cl == child:
                    old_cl = cl

        # Check if there is similar classifier already
        if old_cl is None:
            for cl in self:
                if cl == child:
                    old_cl = cl

        if old_cl is None:
            new_list.append(child)
        else:
            old_cl.increase_quality()
