from agent import Classifier
from agent.acs2 import Constants as c

import logging
from random import random, choice


logger = logging.getLogger(__name__)


class ACS2Utils:

    @staticmethod
    def generate_initial_classifiers(number_of_actions: int = None) -> list:
        """
        Generate a list of default, general classifiers for all
        possible actions.

        :param number_of_actions: number of general classifiers to be generated
        :return: list of classifiers
        """
        if number_of_actions is None:
            number_of_actions = c.NUMBER_OF_POSSIBLE_ACTIONS

        initial_classifiers = []

        for i in range(number_of_actions):
            cl = Classifier()
            cl.action = i
            cl.t = 0

            initial_classifiers.append(cl)

        logger.debug('Generated initial classifiers: %s', initial_classifiers)
        return initial_classifiers

    @staticmethod
    def generate_match_set(classifiers: list, perception: list) -> list:
        """
        Generates a list of classifiers from population that match given
        perception. A list is then stored
        as agents property.

        :param classifiers: list of classifiers
        :param perception: environment perception as list of values
        :return: a list of matching classifiers
        """
        match_set = []

        for cls in classifiers:
            if __class__._does_match(cls, perception):
                match_set.append(cls)

        logger.debug('Generated match set: [%s]', match_set)
        return match_set

    @staticmethod
    def generate_action_set(classifiers: list, action: int) -> list:
        """
        Generates a list of classifiers with matching action.

        :param classifiers: a list of classifiers
        :param action: desired action identifier
        :return: a list of classifiers
        """
        action_set = []

        for cls in classifiers:
            if cls.action == action:
                action_set.append(cls)

        logger.debug('Generated action set: [%s]', action_set)
        return action_set

    @staticmethod
    def choose_action(classifiers: list, epsilon=None) -> int:
        """
        TODO: Chooses action from available classifiers.
        :param classifiers: a list of classifiers
        :param epsilon:
        :return:
        """
        if epsilon is None:
            epsilon = c.EPSILON

        if random() < epsilon:
            all_actions = [i for i in range(c.NUMBER_OF_POSSIBLE_ACTIONS)]
            random_action = choice(all_actions)
            logger.debug('Action chosen: [%d] (randomly)', random_action)
            return random_action
        else:
            best_classifier = classifiers[0]
            dontcare_classifier = [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH
            for cls in classifiers:
                if (cls.effect != dontcare_classifier and
                        cls.fitness() > best_classifier.fitness()):
                    best_classifier = cls

            logger.debug('Action chosen: [%d] (%s)',
                         best_classifier.action, best_classifier)

            return best_classifier.action

    @staticmethod
    def _does_match(cls: Classifier, perception: list) -> bool:
        """
        Check if classifier condition match given perception

        :param cls: classifier object
        :param perception: perception given as list
        :return: True if classifiers can be applied
        for given perception, false otherwise
        """
        if len(perception) != len(cls.condition):
            raise ValueError('Perception and classifier condition '
                             'length is different')

        for i in range(len(perception)):
            if (cls.condition[i] != c.CLASSIFIER_WILDCARD and
                    cls.condition[i] != perception[i]):
                return False

        return True
