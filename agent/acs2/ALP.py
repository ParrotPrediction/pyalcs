from agent.acs2.Classifier import Classifier
from agent.acs2 import Constants as c


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
                newCls = expected_case(classifier, perception)
                was_expected_case += 1
            else:
                newCls = unexpected_case(classifier,
                                         perception,
                                         previous_perception)
                if classifier.q < const.THETA_I:
                    remove(classifier, classifiers)
                    action_set.remove(classifier)

            if newCls is not None:
                newCls.tga = time
                add_alp_classifier(newCls, classifiers, action_set)

            if was_expected_case == 0:
                newCls = cover_triple(previous_perception,
                                      action,
                                      perception,
                                      time)
                add_alp_classifier(newCls, classifiers, action_set)

    @staticmethod
    def _update_application_average(clsf: Classifier, time: int):
        if clsf.exp < 1 / c.BETA:
            clsf.aav += (time - clsf.tga - clsf.aav) / clsf.exp
        else:
            clsf.aav += c.BETA * (time - clsf.tga - clsf.aav)

        # TGA? Should this be in ALP module?
        # Maybe naming convention should be changed
        clsf.tga = time

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
