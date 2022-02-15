from __future__ import annotations

import logging
import random
from itertools import chain
from typing import Optional, List

import lcs.agents.acs as acs
import lcs.agents.acs2er.alp as alp_acs2
import lcs.strategies.anticipatory_learning_process as alp
import lcs.strategies.genetic_algorithms as ga
import lcs.strategies.reinforcement_learning as rl
from lcs import Perception
from lcs.agents.acs2er import Configuration
from . import Classifier


class ClassifiersList(acs.ClassifiersList):

    def __init__(self, *args, oktypes=(Classifier,)) -> None:
        super().__init__(*args, oktypes=oktypes)

    def form_match_set(self, situation: Perception) -> ClassifiersList:
        matching_ls = [cl for cl in self if cl.does_match(situation)]
        return ClassifiersList(*matching_ls)

    def form_action_set(self, action: int) -> ClassifiersList:
        matching = [cl for cl in self if cl.action == action]
        return ClassifiersList(*matching)

    def form_match_set_backwards(self,
                                 situation: Perception) -> ClassifiersList:

        matching = [cl for cl in self if cl.does_match_backwards(situation)]
        return ClassifiersList(*matching)

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

    @staticmethod
    def apply_enhanced_effect_part_check(action_set: ClassifiersList,
                                         new_list: ClassifiersList,
                                         previous_situation: Perception,
                                         time: int,
                                         cfg: Configuration):
        # Create a list of candidates.
        # Every enhanceable classifier is a candidate.
        candidates = [classifier for classifier in action_set
                      if classifier.is_enhanceable()]

        logging.debug(
            "Applying enhanced effect part; number of candidates={}; " +
            "previous situation: {}".format(
                len(candidates), previous_situation))

        # If there are less than 2 candidates, don't do it
        if len(candidates) < 2:
            return

        for candidate in candidates:
            candidates2 = [classifier for classifier in candidates
                           if candidate != classifier]
            if len(candidates2) > 0:
                merger = random.choice(candidates2)
                new_classifier = candidate.merge_with(merger,
                                                      previous_situation,
                                                      time)
                if new_classifier is not None:
                    candidate.reverse_increase_quality()
                    alp.add_classifier(new_classifier, action_set, new_list,
                                       cfg.theta_exp)

        return new_list

    @staticmethod
    def apply_alp(population: ClassifiersList,
                  match_set: ClassifiersList,
                  action_set: ClassifiersList,
                  p0: Perception,
                  action: int,
                  p1: Perception,
                  time: int,
                  theta_exp: int,
                  cfg: Configuration) -> None:
        """
        The Anticipatory Learning Process. Handles all updates by the ALP,
        insertion of new classifiers in pop and possibly matchSet, and
        deletion of inadequate classifiers in pop and possibly matchSet.

        Parameters
        ----------
        population
        match_set
        action_set
        p0: Perception
        action: int
        p1: Perception
        time: int
        theta_exp
        cfg: Configuration

        Returns
        -------

        """
        new_list = ClassifiersList()
        new_cl: Optional[Classifier] = None
        was_expected_case = False
        delete_count = 0

        for cl in action_set:
            cl.increase_experience()
            cl.update_application_average(time)

            if cl.does_anticipate_correctly(p0, p1):
                new_cl = alp_acs2.expected_case(cl, p0, time)
                was_expected_case = True
            else:
                new_cl = alp_acs2.unexpected_case(cl, p0, p1, time)

                if cl.is_inadequate():
                    # Removes classifier from population, match set
                    # and current list
                    delete_count += 1
                    lists = [x for x in [population, match_set, action_set]
                             if x]
                    for lst in lists:
                        lst.safe_remove(cl)

            if new_cl is not None:
                new_cl.tga = time
                alp.add_classifier(new_cl, action_set, new_list, theta_exp)

        if cfg.do_pee:
            ClassifiersList.apply_enhanced_effect_part_check(action_set,
                                                             new_list,
                                                             p0,
                                                             time,
                                                             cfg)

        # No classifier anticipated correctly - generate new one
        if not was_expected_case:
            new_cl = alp_acs2.cover(p0, action, p1, time, cfg)
            alp.add_classifier(new_cl, action_set, new_list, theta_exp)

        # Merge classifiers from new_list into self and population
        action_set.extend(new_list)
        population.extend(new_list)

        if match_set is not None:
            new_matching = [cl for cl in new_list if
                            cl.condition.does_match(p1)]
            match_set.extend(new_matching)

    @staticmethod
    def apply_ga(time: int,
                 population: ClassifiersList,
                 match_set: ClassifiersList,
                 action_set: ClassifiersList,
                 p: Perception,
                 theta_ga: int,
                 mu: float,
                 chi: float,
                 theta_as: int,
                 do_subsumption: bool,
                 theta_exp: int) -> None:

        if ga.should_apply(action_set, time, theta_ga):
            ga.set_timestamps(action_set, time)

            # Select parents
            parent1, parent2 = ga.roulette_wheel_selection(
                action_set, lambda cl: pow(cl.q, 3) * cl.num)

            child1 = Classifier.copy_from(parent1, time)
            child2 = Classifier.copy_from(parent2, time)

            # Execute mutation
            ga.generalizing_mutation(child1, mu)
            ga.generalizing_mutation(child2, mu)

            # Execute cross-over
            if random.random() < chi:
                if child1.effect == child2.effect:
                    ga.two_point_crossover(child1, child2)

                    # Update quality and reward
                    child1.q = child2.q = float(sum([child1.q, child2.q]) / 2)
                    child1.r = child2.r = float(sum([child1.r, child2.r]) / 2)

            child1.q /= 2
            child2.q /= 2

            # We are interested only in classifiers with specialized condition
            unique_children = {cl for cl in [child1, child2]
                               if cl.condition.specificity > 0}

            ga.delete_classifiers(
                population, match_set, action_set,
                len(unique_children), theta_as)

            # check for subsumers / similar classifiers
            for child in unique_children:
                ga.add_classifier(child, p,
                                  population, match_set, action_set,
                                  do_subsumption, theta_exp)

    @staticmethod
    def apply_reinforcement_learning(action_set: ClassifiersList,
                                     reward: int,
                                     p: float,
                                     beta: float,
                                     gamma: float) -> None:
        for cl in action_set:
            rl.update_classifier(cl, reward, p, beta, gamma)

    def get_best_classifier(self) -> Optional[Classifier]:
        anticipated_change_cls = [cl for cl in self if
                                  cl.does_anticipate_change()]

        if len(anticipated_change_cls) > 0:
            random.shuffle(anticipated_change_cls)
            return max(anticipated_change_cls,
                       key=lambda cl: cl.fitness * cl.num)

        return None
