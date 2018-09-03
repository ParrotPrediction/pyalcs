from __future__ import annotations

from typing import Optional

from lcs import TypedList, Perception
from lcs.agents.racs import Configuration
from . import Classifier
from .components.alp import expected_case, unexpected_case, cover


class ClassifierList(TypedList):

    def __init__(self, *args) -> None:
        super().__init__((Classifier,), *args)

    def form_match_set(self, situation: Perception) -> ClassifierList:
        matching = [cl for cl in self if cl.condition.does_match(situation)]
        return ClassifierList(*matching)

    def form_action_set(self, action: int) -> ClassifierList:
        matching = [cl for cl in self if cl.action == action]
        return ClassifierList(*matching)

    def apply_alp(self,
                  p0: Perception,
                  action: int,
                  p1: Perception,
                  time: int,
                  population: ClassifierList,
                  match_set: ClassifierList,
                  cfg: Configuration) -> None:

        new_list = ClassifierList()
        new_cl: Optional[Classifier] = None
        was_expected_case = False
        delete_counter = 0

        for cl in self:
            cl.increase_experience()
            cl.set_alp_timestamp(time)

            if cl.does_anticipate_correctly(p0, p1):
                new_cl = expected_case(cl, p0, time)
                was_expected_case = True
            else:
                new_cl = unexpected_case(cl,
                                         p0,
                                         p1,
                                         time)
                if cl.is_inadequate():
                    delete_counter += 1
                    for lst in [population, match_set, self]:
                        lst.safe_remove(cl)

            if new_cl is not None:
                new_cl.tga = time
                self.add_alp_classifier(new_cl, new_list)

        # No classifier anticipated correctly - generate new one
        if not was_expected_case:
            new_cl = cover(p0, action, p1, time, cfg)
            self.add_alp_classifier(new_cl, new_list)

        # Merge classifiers from new_list into self and population
        self.extend(new_list)
        population.extend(new_list)

        if match_set is not None:
            new_matching = [cl for cl in new_list if
                            cl.condition.does_match(p1)]
            match_set.extend(new_matching)

    def apply_reinforcement_learning(self, reward: int, p: float) -> None:
        """
        Reinforcement Learning. Applies RL according to
        current reinforcement `reward` and back-propagated reinforcement
        `maximum_fitness`.

        Parameters
        ----------
        reward: int
            current reward obtained from the environment
        p: float
            maximum fitness - back-propagated reinforcement
        """
        for cl in self:
            cl.update_reward(reward + cl.cfg.gamma * p)
            cl.update_intermediate_reward(reward)

    def add_alp_classifier(self,
                           child: Classifier,
                           new_list: ClassifierList) -> None:
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
