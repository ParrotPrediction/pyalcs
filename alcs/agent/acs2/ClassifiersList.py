from alcs.agent import Perception
from alcs.agent.acs2 import Classifier
from alcs.agent.acs2 import Constants as c

import logging
from random import random, randint
from copy import copy
from itertools import groupby

logger = logging.getLogger(__name__)


class ClassifiersList(list):
    """
    Represents overall population, match/action sets
    """

    def __init__(self, *args):
        list.__init__(self, *args)

    def append(self, item):
        if not isinstance(item, Classifier):
            raise TypeError("Item should be a Classifier object")

        super(ClassifiersList, self).append(item)

    @classmethod
    def form_match_set(cls, population, situation: Perception):
        return cls([cl for cl in population if cl.condition.does_match(situation)])

    @classmethod
    def form_action_set(cls, population, action: int):
        return cls([cl for cl in population if cl.action == action])

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
        anticipated_change_cls = [cl for cl in self if cl.does_anticipate_change()]

        if len(anticipated_change_cls) > 0:
            best_classifier = max(anticipated_change_cls, key=lambda cl: cl.fitness)

        if best_classifier is not None:
            return best_classifier.action

        return self.choose_random_action()

    def choose_latest_action(self) -> int:
        """
        Chooses latest executed action ("action delay bias")

        :return: chosen action number
        """
        last_executed_cls = None
        number_of_cls_per_action = {i: 0 for i in range(c.NUMBER_OF_POSSIBLE_ACTIONS)}

        if len(self) > 0:
            last_executed_cls = min(self, key=lambda cl: cl.talp)

            self.sort(key=lambda cl: cl.action)
            for _action, _clss in groupby(self, lambda cl: cl.action):
                number_of_cls_per_action[_action] = sum([cl.num for cl in _clss])

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
        knowledge_array = {i: 0 for i in range(c.NUMBER_OF_POSSIBLE_ACTIONS)}
        self.sort(key=lambda cl: cl.action)

        for _action, _clss in groupby(self, lambda cl: cl.action):
            _classifiers = [cl for cl in _clss]

            agg_q = sum(cl.q * cl.num for cl in _classifiers)
            agg_num = sum(cl.num for cl in _classifiers)

            knowledge_array[_action] = agg_q / float(agg_num)

        by_quality = sorted(knowledge_array.items(), key=lambda el: el[1])
        action = by_quality[0][0]

        return action

    @staticmethod
    def choose_random_action() -> int:
        """
        Chooses one of the possible actions in the environment randomly
        :return: random action number
        """
        return randint(0, c.NUMBER_OF_POSSIBLE_ACTIONS - 1)

    def get_maximum_fitness(self) -> float:
        """
        Returns the maximum fitness value amongst those classifiers
        that anticipated a change in environment.

        :return: fitness value
        """
        anticipated_change_cls = [cl for cl in self if cl.does_anticipate_change()]

        if len(anticipated_change_cls) > 0:
            best_cl = max(anticipated_change_cls, key=lambda cl: cl.fitness)
            return best_cl.fitness

        return 0.0

    def set_alp_timestamps(self, time: int) -> None:
        """
        Sets the ALP time stamp to monitor the frequency of application
        and the last application. This method also sets the application
        average parameter.

        :param time: current step
        """
        for cl in self:
            cl.set_alp_timestamp(time)

    def apply_alp(self, previous_situation: Perception, action, situation: Perception, time, population, match_set) -> None:
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
        :return:
        """
        self.set_alp_timestamps(time)

        new_list = ClassifiersList()
        new_cl = None
        found_expected_case = False

        # Because we will be changing classifiers (adding/removing) - we will
        # iterate over the copy of the list
        for cl in copy(self):
            cl.increase_experience()

            if cl.does_anticipate_correctly(previous_situation, situation):
                new_cl = cl.expected_case(previous_situation, time)
                found_expected_case = True
            else:
                new_cl = cl.unexpected_case(previous_situation, situation, time)

                if cl.q < c.THETA_I:
                    # Removes classifier from population, match set
                    # and current list

                    # TODO: p0: validate classifier deletion
                    # This should remove object from memory
                    # and modify all references
                    # from population, match set, self

                    # population.remove(cl)
                    # match_set.remove(cl)
                    # self.remove(cl)
                    # del cl
                    pass

            if new_cl is not None:
                self.insert_alp_offspring(new_cl, new_list)

        if not found_expected_case:
            new_cl = Classifier.cover_triple(previous_situation, action, situation, time)
            self.insert_alp_offspring(new_cl, new_list)

        # Merge classifiers from new_list into self and population
        # I think these classifiers shouldn't be reconsidered in iteration
        # TODO: p1: validate if works properly.
        self.extend(new_list)
        population.extend(new_list)

        if match_set is not None:
            match_set.add_matching_classifiers(new_list, situation)

    def apply_reinforcement_learning(self, rho, p) -> None:
        """
        Reinforcement Learning. Applies RL according to
        current reinforcement `reward` and back-propagated reinforcement
        `maximum_fitness`.

        :param rho: current reward
        :param p: maximum fitness - back-propagated reinforcement
        """
        for cl in self:
            cl.update_reward(rho + c.GAMMA * p)
            cl.update_intermediate_reward(rho)

    def apply_ga(self, time, population, match_set, situation) -> None:
        pass

    def insert_alp_offspring(self, child_cl, new_list) -> bool:
        """
        Looks for subsuming / similar classifiers in the current set.
        If no appropriate classifier was found, the `child_cl` is added to
        `new_list`.

        :param child_cl:
        :param new_list:
        :return: True if an appropriate old classifier was found,
        false otherwise
        """
        # TODO: p4: write tests
        subsumer = self.get_subsumer(child_cl)
        already_created = new_list.get_similar(child_cl)
        already_existed = self.get_similar(child_cl)

        old_cl = subsumer if subsumer is not None else None
        old_cl = already_created if already_created is not None else old_cl
        old_cl = already_existed if already_existed is not None else old_cl

        if old_cl is None:
            new_list.append(child_cl)
            return False
        else:
            old_cl.increase_quality()

        return True

    def add_matching_classifiers(self, lst, situation: Perception):
        """
        Add classifiers matching perception from
        the given list to the current list

        :param lst: another ClassifiersList
        :param situation:
        """
        new_matching = [cl for cl in lst if cl.condition.does_match(situation)]
        self.extend(new_matching)

    def get_subsumer(self, scl: Classifier) -> Classifier:
        """
        Searches the list for the subsuming classifier.
        
        :param scl: 
        :return: subsuming classifier, None otherwise 
        """
        # TODO: p4: write tests
        # Doesnt find any subsumers
        subsumer = None
        subsumers = [cl for cl in self if cl.does_subsume(scl)]

        tmp_subs_lst = []

        for cl in subsumers:
            if subsumer is None:
                subsumer = cl
                tmp_subs_lst.append(subsumer)
            elif cl.is_more_general(subsumer):
                # Another more general subsumer found
                subsumer = cl
                tmp_subs_lst.clear()
                tmp_subs_lst.append(subsumer)
            elif not subsumer.is_more_general(cl):
                # Another subsumer with equal generality was found
                tmp_subs_lst.append(cl)

        # Look through found subsumers and randomly select one
        if len(tmp_subs_lst) > 0:
            current_num = 0
            micro_cls_numerosity = sum(cl.num for cl in tmp_subs_lst)
            select = randint(1, micro_cls_numerosity)

            for cl in tmp_subs_lst:
                current_num += cl.num
                if current_num >= select:
                    return cl

        return subsumer

    def get_similar(self, other: Classifier) -> Classifier:
        """
        Searches for the first similar classifier `other` and returns it.

        :param other: classifier to compare
        :return: first similar classifier, None otherwise
        """
        return next(filter(lambda cl: cl.is_similar(other), self), None)
