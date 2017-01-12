from agent.acs2.Classifier import Classifier
from agent.acs2 import Constants as c


class ALP:

    @staticmethod
    def apply(classifiers: list,
              action: int,
              time: int,
              action_set: list,
              perception: list,
              previous_perception: list):

        was_expected_case = 0

        for cls in action_set:
            cls.exp += 1
            __class__._update_application_average(cls, time)

            if __class__._does_anticipate_correctly(cls,
                                                    perception,
                                                    previous_perception):
                newCls = expected_case(cls, perception)
                was_expected_case += 1
            else:
                newCls = unexpected_case(cls, perception, previous_perception)
                if cls.q < const.THETA_I:
                    remove(cls, classifiers)
                    action_set.remove(cls)

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
    def _update_application_average(cls: Classifier, time: int):
        if cls.exp < 1 / c.BETA:
            cls.aav += (time - cls.tga - cls.aav) / cls.exp
        else:
            cls.aav += c.BETA * (time - cls.tga - cls.aav)

        # TGA? Should this be in ALP module?
        # Maybe naming convention should be changed
        cls.tga = time

    @staticmethod
    def _does_anticipate_correctly(cls: Classifier,
                                   perception: list,
                                   previous_perception: list) -> bool:
        """
        :param cls: given classifier
        :param perception: current perception
        :param previous_perception: previous perception
        :return: True if classifier anticipates correctly, False otherwise
        """
        for i in range(c.CLASSIFIER_LENGTH):
            if cls.effect == c.CLASSIFIER_WILDCARD:
                if previous_perception[i] != perception[i]:
                    return False
            else:
                if (cls.effect[i] != perception[i] or
                        previous_perception[i] == perception[i]):
                    return False

        return True
