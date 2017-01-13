from agent.acs2.Classifier import Classifier
from agent.acs2 import Constants as c

from random import random


class ALP:

    @classmethod
    def apply(cls,
              classifiers: list,
              action: int,
              time: int,
              action_set: list,
              perception: list,
              previous_perception: list):

        was_expected_case = 0

        for classifier in action_set:
            classifier.exp += 1
            cls._update_application_average(classifier, time)

            if cls._does_anticipate_correctly(classifier,
                                              perception,
                                              previous_perception):
                new_classifier = cls._expected_case(classifier, perception)
                was_expected_case += 1
            else:
                new_classifier = cls._unexpected_case(classifier,
                                                      perception,
                                                      previous_perception)
                if classifier.q < c.THETA_I:
                    cls._remove(classifier, classifiers)
                    action_set.remove(classifier)

            if new_classifier is not None:
                new_classifier.tga = time
                cls._add_alp_classifier(new_classifier,
                                        classifiers,
                                        action_set)

        if was_expected_case == 0:
            new_classifier = cls._cover_triple(previous_perception,
                                               perception,
                                               action,
                                               time)
            cls._add_alp_classifier(new_classifier, classifiers, action_set)

    @staticmethod
    def _expected_case(classifier: Classifier,
                       perception: list) -> Classifier:

        diff = get_differences(classifier.mark, perception)

        if diff == [c.CLASSIFIER_WILDCARD] * c.CLASSIFIER_LENGTH:
            classifier.q += c.BETA * (1 - classifier.q)
            return None
        else:
            spec = number_of_spec(classifier.condition)
            spec_new = number_of_spec(diff)
            child = Classifier.copy_from(classifier)

            if spec == c.U_MAX:
                remove_random_spec_att(child.condition)
                spec -= 1

                while spec + spec_new > c.BETA:
                    if spec > 0 and random() < 0.5:
                        remove_random_spec_att(child.condition)
                        spec -= 1
                    else:
                        remove_random_spec_att(diff)
                        spec_new -= 1
            else:
                while spec + spec_new > c.BETA:
                    remove_random_spec_att(diff)
                    spec_new -= 1

            child.condition = diff
            child.exp = 1

            if child.q < 0.5:
                child.q = 0.5

            return child

    @staticmethod
    def _unexpected_case(classifier: Classifier,
                         perception: list,
                         previous_perception: list) -> Classifier:

        classifier.q = classifier.q - c.BETA * classifier.q
        classifier.mark = previous_perception

        for i in range(len(perception)):
            if classifier.effect[i] != c.CLASSIFIER_WILDCARD:
                if (classifier.effect[i] != previous_perception[i] or
                        previous_perception[i] != perception[i]):
                    return None

        child = Classifier.copy_from(classifier)

        for i in random(len(perception)):
            if (classifier.effect[i] == c.CLASSIFIER_WILDCARD and
                    previous_perception[i] != perception[i]):
                child.condition[i] = previous_perception[i]
                child.effect[i] = perception[i]

        if classifier.q < 0.5:
            classifier.q = 0.5

        child.exp = 1

        return child

    @staticmethod
    def _update_application_average(cla: Classifier, time: int):
        if cla.exp < 1 / c.BETA:
            cla.aav += (time - cla.tga - cla.aav) / cla.exp
        else:
            cla.aav += c.BETA * (time - cla.tga - cla.aav)

        # TGA? Should this be in ALP module?
        # Maybe naming convention should be changed
        cla.tga = time

    @staticmethod
    def _add_alp_classifier(classifier: Classifier,
                            classifiers: list,
                            action_set: list) -> None:

        old_classifier = None

        for cla in action_set:
            if cla.is_subsumer(classifier):
                if old_classifier is None or cla.is_more_general(classifier):
                    old_classifier = cla

        if old_classifier is None:
            for cla in action_set:
                if cla == classifier:
                    old_classifier = cla

        if old_classifier is None:
            classifiers.append(classifier)
            action_set.append(classifier)
        else:
            old_classifier.q += c.BETA * (1 - old_classifier.q)

    @staticmethod
    def _cover_triple(previous_perception: list,
                      perception: list,
                      action: int,
                      time: int) -> Classifier:

        child = Classifier()

        for i in range(len(perception)):
            if previous_perception[i] != perception[i]:
                child.condition[i] = previous_perception[i]
                child.effect[i] = perception[i]

        child.action = action
        child.alp = time
        child.tga = time
        child.t = time

        return child

    @staticmethod
    def _does_anticipate_correctly(classifier: Classifier,
                                   perception: list,
                                   previous_perception: list) -> bool:
        """
        :param classifier: given classifier
        :param perception: current perception
        :param previous_perception: previous perception
        :return: True if classifier anticipates correctly, False otherwise
        """
        for i in range(c.CLASSIFIER_LENGTH):
            if classifier.effect == c.CLASSIFIER_WILDCARD:
                if previous_perception[i] != perception[i]:
                    return False
            else:
                if (classifier.effect[i] != perception[i] or
                        previous_perception[i] == perception[i]):
                    return False

        return True

    @staticmethod
    def _remove(classifier: Classifier, classifiers: list) -> bool:
        """
        Removes classifier with the same condition, action
        and effect part from the classifiers list

        :param classifier classifier to be removed
        :param classifiers classifiers list
        :return True is classifier was removed, False otherwise
        """
        for i in range(len(classifiers)):
            # TODO: maybe move into ACS2Utils class
            # TODO: maybe __eq__ in Classifier could be used ...
            if (classifier.condition == classifiers[i].condition and
                    classifier.action == classifiers[i].action and
                    classifier.effect == classifiers[i].effect):
                del classifiers[i]
                return True

        return False
