from alcs.agent import Perception
from alcs.agent.acs3 import Classifier, Action
from alcs.agent.acs3 import Constants as c

from random import random, randint
from itertools import groupby


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
    def form_action_set(cls, population, action: Action):
        return cls([cl for cl in population if cl.action == action])

    def choose_action(self, epsilon: float) -> Action:
        """
        Chooses action according to epsilon greedy policy
        :param epsilon: probability of executing exploration path
        :return: number of chosen action
        """
        if random() < epsilon:
            return self.choose_explore_action()

        return self.choose_best_fitness_action()

    def choose_explore_action(self, pb: float = 0.5) -> Action:
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

    def choose_best_fitness_action(self) -> Action:
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

    def choose_latest_action(self) -> Action:
        """
        Chooses latest executed action ("action delay bias")

        :return: chosen action
        """
        last_executed_cls = None
        number_of_cls_per_action = {i: 0 for i in range(c.NUMBER_OF_POSSIBLE_ACTIONS)}

        if len(self) > 0:
            last_executed_cls = min(self, key=lambda cl: cl.talp)

            self.sort(key=lambda cl: cl.action)
            for _action, _clss in groupby(self, lambda cl: cl.action):
                number_of_cls_per_action[_action.action] = sum([cl.num for cl in _clss])

        # If there are some actions with no classifiers - select them
        for action, nCls in number_of_cls_per_action.items():
            if nCls == 0:
                return Action(action)

        # Otherwise return the action of the last executed classifier
        return last_executed_cls.action

    def choose_action_from_knowledge_array(self) -> Action:
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

            knowledge_array[_action.action] = agg_q / float(agg_num)

        by_quality = sorted(knowledge_array.items(), key=lambda el: el[1])
        action = by_quality[0][0]

        return Action(action)

    @staticmethod
    def choose_random_action() -> Action:
        """
        Chooses one of the possible actions in the environment randomly
        :return: random action
        """
        return Action(randint(0, c.NUMBER_OF_POSSIBLE_ACTIONS - 1))

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
        new_cl = None

        for cl in self:
            cl.increase_experience()

            if cl.does_anticipate_correctly(previous_situation, situation):
                pass
            else:
                pass

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
