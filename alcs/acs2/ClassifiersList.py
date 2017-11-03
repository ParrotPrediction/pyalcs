import logging
from itertools import groupby, chain
from random import random, randint, choice, sample

from alcs.acs2 import Classifier

from alcs.acs2 import ACS2Configuration
from alcs import Perception

DO_SUBSUMPTION = True

logger = logging.getLogger(__name__)


class ClassifiersList(list):
    """
    Represents overall population, match/action sets
    """

    def __init__(self, seq=(), cfg=None):
        self.cfg = cfg or ACS2Configuration.default()
        list.__init__(self, seq or [])

    def append(self, item):
        if not isinstance(item, Classifier):
            raise TypeError("Item should be a Classifier object")
        # print(type(self))
        super(ClassifiersList, self).append(item)

    @classmethod
    def form_match_set(cls, population, situation: Perception):
        return cls([cl for cl in population
                    if cl.condition.does_match(situation)])

    @classmethod
    def form_action_set(cls, population, action: int):
        return cls([cl for cl in population
                    if cl.action == action])

    @staticmethod
    def _remove_classifier(population, cl: Classifier):
        """
        Searches the list and removes classifier
        :param cl: classifier to remove
        """
        # TODO p4: write test
        if population is not None and cl in population:
            population.remove(cl)

    def expand(self):
        """
        Returns an array containing all micro-classifiers
        :return: a list of all classifiers
        """
        list2d = [[cl] * cl.num for cl in self]
        return list(chain.from_iterable(list2d))

    def choose_action(self, epsilon: float) -> int:
        """
        Chooses action according to epsilon greedy policy
        :param epsilon: probability of executing exploration path
        :return: number of chosen action
        """
        if random() < epsilon:
            logger.debug("Exploration path")
            return self.choose_explore_action()

        logger.debug("Exploitation path")
        return self.choose_best_fitness_action()

    def choose_explore_action(self, pb: float = 0.5) -> int:
        """
        Chooses action according to current exploration policy

        :param pb: probability of biased exploration
        :return: action to be executed
        """
        if random() < pb:
            # We are in the biased exploration
            if random() < 0.5:
                return self.choose_latest_action()
            else:
                return self.choose_action_from_knowledge_array()

        return self.choose_random_action()

    def choose_best_fitness_action(self) -> int:
        """
        Chooses best action according to fitness. If there is no classifier
        in list (or none is predicting change) than a random action is returned

        :return: action from the best classifier
        """
        best_classifier = None
        anticipated_change_cls = [cl for cl in self
                                  if cl.does_anticipate_change()]

        if len(anticipated_change_cls) > 0:
            best_classifier = max(anticipated_change_cls,
                                  key=lambda cl: cl.fitness)

        if best_classifier is not None:
            return best_classifier.action

        return self.choose_random_action()

    def choose_latest_action(self) -> int:
        """
        Chooses latest executed action ("action delay bias")

        :return: chosen action number
        """
        last_executed_cls = None
        number_of_cls_per_action = \
            {i: 0 for i in range(self.cfg.number_of_possible_actions)}

        if len(self) > 0:
            last_executed_cls = min(self, key=lambda cl: cl.talp)

            self.sort(key=lambda cl: cl.action)
            for _action, _clss in groupby(self, lambda cl: cl.action):
                number_of_cls_per_action[_action] = \
                    sum([cl.num for cl in _clss])

        # If there are some actions with no classifiers - select them
        for action, nCls in number_of_cls_per_action.items():
            if nCls == 0:
                return action

        # Otherwise return the action of the last executed classifier
        return last_executed_cls.action

    def choose_action_from_knowledge_array(self) -> int:
        """
        Creates 'knowledge array' that represents the average quality of the
        anticipation for each action in the current list. Chosen is
        the action, ACS2 knows least about the consequences.

        :return: chosen action
        """
        knowledge_array = {i: 0
                           for i in range(self.cfg.number_of_possible_actions)}
        self.sort(key=lambda cl: cl.action)

        for _action, _clss in groupby(self, lambda cl: cl.action):
            _classifiers = [cl for cl in _clss]

            agg_q = sum(cl.q * cl.num for cl in _classifiers)
            agg_num = sum(cl.num for cl in _classifiers)

            knowledge_array[_action] = agg_q / float(agg_num)

        by_quality = sorted(knowledge_array.items(), key=lambda el: el[1])
        action = by_quality[0][0]

        return action

    def choose_random_action(self) -> int:
        """
        Chooses one of the possible actions in the environment randomly
        :return: random action number
        """
        return randint(0, self.cfg.number_of_possible_actions - 1)

    def get_maximum_fitness(self) -> float:
        """
        Returns the maximum fitness value amongst those classifiers
        that anticipated a change in environment.

        :return: fitness value
        """
        anticipated_change_cls = [cl for cl in self
                                  if cl.does_anticipate_change()]

        if len(anticipated_change_cls) > 0:
            best_cl = max(anticipated_change_cls, key=lambda cl: cl.fitness)
            return best_cl.fitness

        return 0.0

    def apply_alp(self,
                  previous_situation: Perception,
                  action: int,
                  situation: Perception,
                  time: int,
                  population,
                  match_set) -> None:
        """
        The Anticipatory Learning Process. Handles all updates by the ALP,
        insertion of new classifiers in pop and possibly matchSet, and
        deletion of inadequate classifiers in pop and possibly matchSet.

        :param previous_situation:
        :param action:
        :param situation:
        :param time:
        :param population:
        :param match_set:
        """
        new_list = ClassifiersList(cfg=self.cfg)
        new_cl = None
        was_expected_case = False
        delete_count = 0

        # Because we will be changing classifiers (adding/removing) - we will
        # iterate over the copy of the list
        for cl in self:
            cl.increase_experience()
            cl.set_alp_timestamp(time)

            if cl.does_anticipate_correctly(previous_situation, situation):
                new_cl = cl.expected_case(previous_situation, time)
                was_expected_case = True
            else:
                new_cl = cl.unexpected_case(previous_situation,
                                            situation,
                                            time)

                if cl.is_inadequate():
                    # Removes classifier from population, match set
                    # and current list
                    delete_count += 1
                    for lst in [population, match_set, self]:
                        __class__._remove_classifier(lst, cl)

            if new_cl is not None:
                new_cl.tga = time
                self.add_alp_classifier(new_cl, new_list)

        # No classifier anticipated correctly - generate new one
        if not was_expected_case:
            new_cl = Classifier.cover_triple(previous_situation,
                                             action,
                                             situation,
                                             time)
            self.add_alp_classifier(new_cl, new_list)

        # Merge classifiers from new_list into self and population
        self.extend(new_list)
        population.extend(new_list)

        if match_set is not None:
            new_matching = [cl for cl in new_list if
                            cl.condition.does_match(situation)]
            match_set.extend(new_matching)

    def apply_reinforcement_learning(self, rho, p) -> None:
        """
        Reinforcement Learning. Applies RL according to
        current reinforcement `reward` and back-propagated reinforcement
        `maximum_fitness`.

        :param rho: current reward
        :param p: maximum fitness - back-propagated reinforcement
        """
        for cl in self:
            cl.update_reward(rho + self.cfg.gamma * p)
            cl.update_intermediate_reward(rho)

    def apply_ga(self, time, population, match_set, situation,
                 randomfunc=random, samplefunc=sample) -> None:
        if self.should_apply_ga(time):
            self.set_ga_timestamp(time)

            parent1, parent2 = self.select_parents(randomfunc=randomfunc)

            child1 = Classifier.copy_from(parent1, time)
            child2 = Classifier.copy_from(parent2, time)

            child1.mutate(randomfunc=randomfunc)
            child2.mutate(randomfunc=randomfunc)

            if randomfunc() < self.cfg.chi:
                if child1.effect == child2.effect:
                    child1.crossover(child2, samplefunc=samplefunc)

            child1.q /= 2
            child2.q /= 2

            children = [child for child in [child1, child2]
                        if child.condition.specificity > 0]

            # if two classifiers are identical, leave only one
            if len(children) == 2 and children[0].is_similar(children[1]):
                children = [children[0]]

            self.delete_ga_classifiers(population, match_set,
                                       len(children), randomfunc=randomfunc)

            # check for subsumers / similar classifiers
            for child in children:
                self.add_ga_classifier(child, match_set, population)

    def add_ga_classifier(self, child, match_set, population):
        """
        Find subsumer/similar classifier, if present - increase its numerosity,
        else add this new classifier
        :param child: new classifier to add
        :param match_set:
        :param population:
        :return:
        """
        if child.condition.specificity != 0:
            old_cl = self.find_old_classifier(child, match_set)

            if old_cl is None:
                # TODO  need to add to self?
                # self.append(child)  # ?
                population.append(child)
                if match_set is not None:
                    match_set.append(child)
            else:
                # TODO in C++ code, in case old_cl was found as subsumer,
                # its numerosity is increased anyway
                if not old_cl.is_marked():
                    old_cl.num += 1

    def add_alp_classifier(self, child, new_list):
        """
        Looks for subsuming / similar classifiers in the current set.
        If no appropriate classifier was found, the `child_cl` is added to
        `new_list`.

        :param child:
        :param new_list:
        :return: True if an appropriate old classifier was found,
        false otherwise
        """
        # TODO: p0: write tests
        old_cl = None

        # Look if there is a classifier that subsumes the insertion
        # candidate
        for classifier in self:
            if classifier.does_subsume(child):
                if old_cl is None or classifier.is_more_general(old_cl):
                    old_cl = classifier

                    # Check if any similar classifier wasn't in this
                    # ALP application
        if old_cl is None:
            for classifier in new_list:
                if classifier.is_similar(child):
                    old_cl = classifier

        # Check if there is similar classifier already
        if old_cl is None:
            for classifier in self:
                if classifier.is_similar(child):
                    old_cl = classifier

        if old_cl is None:
            new_list.append(child)
        else:
            old_cl.increase_quality()

    def get_similar(self, other: Classifier) -> Classifier:
        """
        Searches for the first similar classifier `other` and returns it.

        :param other: classifier to compare
        :return: first similar classifier, None otherwise
        """
        return next(filter(lambda cl: cl.is_similar(other), self), None)

    def should_apply_ga(self, time):
        """
        Checks the average last GA application to determine if a GA
        should be applied.If no classifier is in the current set,
        no GA is applied!
        :param time:
        :return:
        """
        overall_time = sum(cl.tga * cl.num for cl in self)
        overall_num = self.overall_numerosity()

        # print("Overall numerosity: %d" % overall_num)

        if overall_num == 0:
            return False

        if time - overall_time / overall_num > self.cfg.theta_ga:
            # print("Shoud apply GA!")
            return True

        return False

    def overall_numerosity(self):
        return sum(cl.num for cl in self)

    def set_ga_timestamp(self, time):
        """
        Sets the GA time stamps to the current time to control
        the GA application frequency.
        :param time:
        :return:
        """
        for cl in self:
            cl.tga = time

    def select_parents(self, randomfunc=random):
        """
        Select two parents for the GA with roulette-wheel selection.
        """
        parent1, parent2 = None, None

        q_sum = sum(cl.q3num() for cl in self)

        q_sel1 = randomfunc() * q_sum
        q_sel2 = randomfunc() * q_sum

        q_sel1, q_sel2 = sorted([q_sel2, q_sel1])

        q_counter = 0.0
        for cl in self:
            q_counter += cl.q3num()
            if parent1 is None and q_counter > q_sel1:
                parent1 = cl
            if q_counter > q_sel2:
                parent2 = cl
                break

        return parent1, parent2

    def delete_ga_classifiers(self, population, match_set, child_no,
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

    def delete_a_classifier(self, match_set, population, randomfunc=random):
        """ Delete one classifier from a population """
        if len(population) == 0:   # Nothing to remove
            return None
        cl_del = self.select_classifier_to_delete(randomfunc=randomfunc)
        if cl_del is not None:
            if cl_del.num > 1:
                cl_del.num -= 1
            else:
                # Removes classifier from population, match set
                # and current list
                for lst in [self, population, match_set]:
                    if lst is not None:
                        __class__._remove_classifier(lst, cl_del)

    def select_classifier_to_delete(self, randomfunc=random):
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
                                   cl_to_delete: Classifier) -> Classifier:
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

    @staticmethod
    def find_old_classifier(cl: Classifier, existing_classifiers):
        if existing_classifiers is None:
            return None

        old_cl = None

        if DO_SUBSUMPTION:
            old_cl = ClassifiersList.find_subsumer(cl, existing_classifiers)

        if old_cl is None:
            old_cl = ClassifiersList.find_similar_classifier(
                cl, existing_classifiers)

        return old_cl

    @staticmethod
    def find_similar_classifier(cl: Classifier,
                                existing_classifiers) -> Classifier:
        return existing_classifiers.get_similar(cl)

    @staticmethod
    def find_subsumer(cl: Classifier,
                      existing_classifiers,
                      choice_func=choice) -> Classifier:
        subsumer = None
        most_general_subsumers = []
        for classifier in existing_classifiers:
            if classifier.does_subsume(cl):
                if subsumer is None:
                    subsumer = classifier
                    most_general_subsumers = [subsumer]
                elif classifier.is_more_general(subsumer):
                    subsumer = classifier
                    most_general_subsumers = [subsumer]
                elif subsumer.is_equally_general(classifier):
                    most_general_subsumers.append(classifier)  # !

        return choice_func(most_general_subsumers) \
            if most_general_subsumers else None
