from __future__ import annotations

from itertools import chain
from random import random, choice, sample
from typing import Optional, List

from lcs import Perception, TypedList
from lcs.agents.acs2.components.alp import expected_case, unexpected_case, \
    cover
from lcs.agents.acs2.components.genetic_algorithm \
    import mutate, two_point_crossover
from lcs.strategies.genetic_algorithms import roulette_wheel_selection
from . import Classifier, Configuration


class ClassifiersList(TypedList):
    """
    Represents overall population, match/action sets
    """

    def __init__(self, *args, cfg: Configuration) -> None:
        self.cfg = cfg
        super().__init__((Classifier,), *args)

    def form_match_set(self,
                       situation: Perception,
                       cfg: Configuration) -> ClassifiersList:
        matching = [cl for cl in self if cl.condition.does_match(situation)]
        return ClassifiersList(*matching, cfg=cfg)

    def form_match_set_backwards(self,
                                 situation: Perception,
                                 cfg: Configuration):
        matching = [cl for cl in self if cl.does_match_backwards(situation)]
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
                new_cl = expected_case(cl, p0, time)
                was_expected_case = True
            else:
                new_cl = unexpected_case(cl, p0, p1, time)

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
            new_cl = cover(p0, action, p1, time, self.cfg)
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

    def apply_ga(self,
                 time: int,
                 population: ClassifiersList,
                 match_set: ClassifiersList,
                 situation: Perception,
                 randomfunc=random,
                 samplefunc=sample) -> None:

        if self.should_apply_ga(time):
            self.set_ga_timestamp(time)

            # Select parents
            parent1, parent2 = roulette_wheel_selection(
                self, lambda cl: pow(cl.q, 3) * cl.num)

            child1 = Classifier.copy_from(parent1, time)
            child2 = Classifier.copy_from(parent2, time)

            mutate(child1, child1.cfg.mu, randomfunc=randomfunc)
            mutate(child2, child2.cfg.mu, randomfunc=randomfunc)

            if randomfunc() < self.cfg.chi:
                if child1.effect == child2.effect:
                    two_point_crossover(child1, child2, samplefunc=samplefunc)

                    # Update quality and reward
                    # TODO: check if needed
                    child2.q = float(sum([child1.q, child2.q]) / 2)
                    child2.r = float(sum([child1.r, child2.r]) / 2)

            child1.q /= 2
            child2.q /= 2

            children = [child for child in [child1, child2]
                        if child.condition.specificity > 0]

            # if two classifiers are identical, leave only one
            unique_children = set(children)

            self.delete_ga_classifiers(population, match_set,
                                       len(unique_children),
                                       randomfunc=randomfunc)

            # check for subsumers / similar classifiers
            for child in unique_children:
                self.add_ga_classifier(child, match_set, population)

    def add_ga_classifier(self,
                          child: Classifier,
                          match_set: ClassifiersList,
                          population: ClassifiersList):
        """
        Find subsumer/similar classifier, if present - increase its numerosity,
        else add this new classifier
        :param child: new classifier to add
        :param match_set:
        :param population:
        :return:
        """
        old_cl = self.find_old_classifier(child)

        if old_cl is None:
            self.append(child)
            population.append(child)
            if match_set is not None:
                match_set.append(child)
        else:
            if not old_cl.is_marked():
                old_cl.num += 1

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

    def get_similar(self, other: Classifier) -> Optional[Classifier]:
        """
        Searches for the first similar classifier `other` and returns it.

        Parameters
        ----------
        other: Classifier
            classifier to compare
        Returns
        -------
        Optional[Classifier]
            classifier (with the same condition, action, effect),
            None otherwise
        """
        return next(filter(lambda cl: cl == other, self), None)

    def should_apply_ga(self, time: int):
        """
        Checks the average last GA application to determine if a GA
        should be applied.If no classifier is in the current set,
        no GA is applied!
        :param time:
        :return:
        """
        overall_time = sum(cl.tga * cl.num for cl in self)
        overall_num = self.overall_numerosity()

        if overall_num == 0:
            return False

        if time - overall_time / overall_num > self.cfg.theta_ga:
            return True

        return False

    def overall_numerosity(self):
        return sum(cl.num for cl in self)

    def set_ga_timestamp(self, time: int):
        """
        Sets the GA time stamps to the current time to control
        the GA application frequency.
        :param time:
        :return:
        """
        for cl in self:
            cl.tga = time

    def delete_ga_classifiers(self,
                              population: ClassifiersList,
                              match_set: ClassifiersList,
                              child_no: int,
                              randomfunc=random):
        """
        Deletes classifiers in the set to keep the size THETA_AS.
        Also considers that still childNo classifiers are added by the GA.
        :param randomfunc:
        :param population:
        :param match_set:
        :param child_no: number of classifiers that will be inserted
        :return:
        """
        del_no = self.overall_numerosity() + child_no - self.cfg.theta_as
        if del_no <= 0:
            # There is still room for more classifiers
            return

        # print("GA: requested to delete: %d classifiers", del_no)
        for _ in range(0, del_no):
            self.delete_a_classifier(
                match_set, population, randomfunc=randomfunc)

    def delete_a_classifier(self,
                            match_set: ClassifiersList,
                            population: ClassifiersList,
                            randomfunc=random):
        """ Delete one classifier from a population """
        if len(population) == 0:  # Nothing to remove
            return None
        cl_del = self.select_classifier_to_delete(randomfunc=randomfunc)
        if cl_del is not None:
            if cl_del.num > 1:
                cl_del.num -= 1
            else:
                # Removes classifier from population, match set
                # and current list
                lists = [x for x in [population, match_set, self] if x]
                for lst in lists:
                    lst.safe_remove(cl_del)

    def select_classifier_to_delete(self, randomfunc=random) -> \
            Optional[Classifier]:

        if len(self) == 0:
            return None

        cl_del = None
        while cl_del is None:  # We must delete at least one
            for cl in self.expand():
                if randomfunc() < 1. / 3.:
                    if cl_del is None:
                        cl_del = cl
                    else:
                        cl_del = self.select_preferred_to_delete(cl, cl_del)
        return cl_del

    @staticmethod
    def select_preferred_to_delete(cl: Classifier,
                                   cl_to_delete: Classifier) -> \
            Classifier:

        if cl.q - cl_to_delete.q < -0.1:
            cl_to_delete = cl
            return cl_to_delete

        if abs(cl.q - cl_to_delete.q) <= 0.1:
            if cl.is_marked() and not cl_to_delete.is_marked():
                cl_to_delete = cl
            elif cl.is_marked or not cl_to_delete.is_marked():
                if cl.tav > cl_to_delete.tav:
                    cl_to_delete = cl
        return cl_to_delete

    def find_old_classifier(self, cl: Classifier):
        old_cl = None

        if self.cfg.do_subsumption:
            old_cl = self.find_subsumer(cl)

        if old_cl is None:
            old_cl = self.get_similar(cl)

        return old_cl

    def find_subsumer(self, cl: Classifier, choice_func=choice) -> \
            Classifier:

        subsumer = None
        most_general_subsumers: List[Classifier] = []

        for classifier in self:
            if classifier.does_subsume(cl):
                if subsumer is None:
                    subsumer = classifier
                    most_general_subsumers = [subsumer]
                elif classifier.is_more_general(subsumer):
                    subsumer = classifier
                    most_general_subsumers = [subsumer]
                elif not subsumer.is_more_general(classifier):
                    most_general_subsumers.append(classifier)  # !

        return choice_func(most_general_subsumers) \
            if most_general_subsumers else None

    def exists_classifier(self, previous_situation, action, situation,
                          quality):
        """
        Returns True if there is a classifier in this list with a quality
        higher than 'quality' that matches previous_situation,
        specifies action, and predicts situation.
        Returns False otherwise.
        :param previous_situation:
        :param action:
        :param situation:
        :param quality:
        :return:
        """
        for cl in self:
            if cl.q > quality and cl.does_match(previous_situation) \
                and cl.action == action \
                and cl.does_anticipate_correctly(previous_situation,
                                                 situation):
                return True
        return False

    def get_quality_classifiers_list(self, quality, cfg=None):
        """
        Constructs classifier list out of a list with q > quality.
        :param quality:
        :param cfg:
        :return: ClassifiersList with only quality classifiers.
        """
        listp = ClassifiersList(cfg=cfg)
        for item in self:
            if item.q > quality:
                listp.append(item)
        return listp

    def search_goal_sequence(self, start, goal):
        """
        Searches a path from start to goal using a bidirectional method in the
        environmental model (i.e. the list of reliable classifiers).
        :param start: Perception
        :param goal: Perception
        :return: Sequence of actions
        """
        reliable_classifiers = self. \
            get_quality_classifiers_list(quality=self.cfg.theta_r,
                                         cfg=self.cfg)

        if len(reliable_classifiers) < 1:
            return self._empty_sequence()

        self.forward_classifiers = []
        self.backward_classifiers = []

        self.forward_perceptions = []
        self.backward_perceptions = []

        max_depth = 6
        forward_size = 1
        backward_size = 1
        forward_point = 0
        backward_point = 0
        action_sequence = None

        self.forward_perceptions.append(Perception(start))
        self.backward_perceptions.append(Perception(goal))

        for depth in range(0, max_depth):
            # forward step
            action_sequence, forward_size_new = self. \
                _search_one_forward_step(reliable_classifiers, forward_size,
                                         forward_point)
            forward_point = forward_size
            forward_size = forward_size_new

            if action_sequence is not None:
                return action_sequence

            # backwards step
            action_sequence, backward_size_new = self. \
                _search_one_backward_step(reliable_classifiers, backward_size,
                                          backward_point)
            backward_point = backward_size
            backward_size = backward_size_new

            if action_sequence is not None:
                return action_sequence

        # depth limit was reached -> return empty action sequence
        return self._empty_sequence()

    def _search_one_forward_step(self, reliable_classifiers, forward_size,
                                 forward_point):
        """
        Serches one step forward in the reliable_classifiers classifier list.
        Returns None if nothing was found so far, a sequence with a -1 element
        if the search failed completely
        (which is the case if the allowed array size of 10000 is reached),
        or the sequence if one was found.
        :param reliable_classifiers: ClassifiersList
        :param forward_size: int
        :param forward_point: int
        :return: act sequence and new forward_size
        """
        size = forward_size
        for i in range(forward_point, forward_size):
            match_forward = reliable_classifiers. \
                form_match_set(situation=self.forward_perceptions[i],
                               cfg=self.cfg)
            for match_set_element in match_forward:
                anticipation = match_set_element. \
                    get_best_anticipation(self.forward_perceptions[i])
                if self._does_contain_state(self.forward_perceptions,
                                            anticipation) is None:
                    # state not detected forward -> search in backwards
                    backward_sequence_idx = self. \
                        _does_contain_state(self.backward_perceptions,
                                            anticipation)
                    if backward_sequence_idx is None:
                        # state neither detected backwards
                        self.forward_perceptions.append(anticipation)
                        self.forward_classifiers.append(
                            self._form_new_classifiers(
                                self.forward_classifiers, i,
                                match_set_element))
                        size += 1
                        if size > 10001:
                            # logging.debug("Arrays are full")
                            return self._empty_sequence(), size
                    else:
                        # sequence found
                        return self. \
                            _sequence_found_forwards(
                                i, backward_sequence_idx,
                                match_set_element), size
        return None, size

    def _search_one_backward_step(self, reliable_classifiers, backward_size,
                                  backward_point):
        """
        Searches one step backward in the reliable_classifiers classifiers list
        Returns None if nothing was found so far, a sequence with a -1 element
        if the search failed completely
        (which is the case if the allowed array size of 10000 is reached),
        or the sequence if one was found.
        :param reliable_classifiers: ClassifiersList
        :param backward_size: int
        :param backward_point: int
        :return: act sequence and new backward_size
        """
        size = backward_size
        for i in range(backward_point, backward_size):
            match_backward = reliable_classifiers. \
                form_match_set_backwards(
                    situation=self.backward_perceptions[i],
                    cfg=self.cfg)
            for match_set_el in match_backward:
                anticipation = match_set_el. \
                    get_backwards_anticipation(self.backward_perceptions[i])
                if anticipation is not None and \
                    self._does_contain_state(self.backward_perceptions,
                                             anticipation) is None:
                    # Backwards anticipation was formable but
                    # not detected backwards
                    forward_sequence_idx = self. \
                        _does_contain_state(self.forward_perceptions,
                                            anticipation)
                    if forward_sequence_idx is None:
                        self.backward_perceptions.append(anticipation)
                        self.backward_classifiers.append(
                            self._form_new_classifiers(
                                self.backward_classifiers, i, match_set_el))
                        size += 1
                        if size > 10001:
                            # logging.debug("Arrays are full")
                            return self._empty_sequence(), size
                    else:
                        return self. \
                            _sequence_found_backwards(
                                i, forward_sequence_idx,
                                match_set_el), size
        return None, size

    def _form_new_classifiers(self, classifiers_lists, i, match_set_el):
        """
        Executes actions after sequence was not detected.
        :param classifiers_lists: list of ClassifiersLists
        :param i: int
        :param match_set_el: Classifier
        :return: new size of classifiers
        """
        if i > 0:
            new_classifiers = ClassifiersList(cfg=self.cfg)
            new_classifiers.extend(classifiers_lists[i - 1])
        else:
            new_classifiers = ClassifiersList(cfg=self.cfg)
        new_classifiers.append(match_set_el)
        return new_classifiers

    def _sequence_found_forwards(self, i, backward_sequence_idx, match_set_el):
        """
        Forms sequence when it was found forwards.
        :param i:
        :param backward_sequence_idx:
        :param match_set_el: Classifier
        :return: act sequence
        """
        # count sequence size
        sequence_size = 0
        if i > 0:
            sequence_size += len(self.forward_classifiers[i - 1])
        if backward_sequence_idx > 0:
            sequence_size += len(
                self.backward_classifiers[backward_sequence_idx - 1])
        sequence_size += 1

        # construct sequence
        act_seq = [-1] * sequence_size
        j = 0
        if i > 0:
            for j, cl in enumerate(self.forward_classifiers[i - 1]):
                act_seq[len(self.forward_classifiers[i - 1]) - j - 1] \
                    = cl.action
            j += 1
        act_seq[j] = match_set_el.action
        j += 1
        if backward_sequence_idx > 0:
            for k, cl in enumerate(
                    self.backward_classifiers[backward_sequence_idx - 1]):
                act_seq[k + j] = cl.action
        return act_seq

    def _sequence_found_backwards(self, i, forward_sequence_idx, match_set_el):
        """
        Forms sequence when it was found backwards.
        :param i: int
        :param forward_sequence_idx: int
        :param match_set_el: Classifier
        :return: act sequence
        """
        # count sequence size
        sequence_size = 0
        if i > 0:
            sequence_size += len(self.backward_classifiers[i - 1])
        if forward_sequence_idx > 0:
            sequence_size += len(
                self.forward_classifiers[forward_sequence_idx - 1])
        sequence_size += 1

        # construct sequence
        act_seq = [-1] * sequence_size
        j = 0
        if forward_sequence_idx > 0:
            for j, cl in enumerate(
                    self.forward_classifiers[forward_sequence_idx - 1]):
                act_seq[
                    len(self.
                        forward_classifiers[
                            forward_sequence_idx - 1]) - j - 1] \
                    = cl.action
            j += 1
        act_seq[j] = match_set_el.action
        j += 1
        if i > 0:
            for k, cl in enumerate(self.backward_classifiers[i - 1]):
                act_seq[k + j] = cl.action
        return act_seq

    @staticmethod
    def _does_contain_state(perceptions, state):
        """
        Returns the position in the perception list where 'state' is stored or
        None if state is not found
        :param perceptions: Perception list
        :param state: Perception
        :return:
        """
        for i, percept in enumerate(perceptions):
            if percept == state:
                return i
        return None

    @staticmethod
    def _empty_sequence():
        """
        Returns empty sequence.
        This function might be deleted in later versions of code, but it is
        useful if we decide to change the definition
        of an empty sequence (it used to be [-1], because it was more alike
        the original code in C++).
        :return: empty action sequence
        """
        return []
