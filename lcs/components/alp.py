from random import random, randint

from lcs import Perception
from ..acs2 import Classifier


def expected_case(cl: Classifier,
                  perception: Perception,
                  time: int) -> Classifier:
    """
    Controls the expected case of a classifier. If the classifier
    is to specific it tries to add some randomness to it by
    generalizing some attributes.

    :param cl:
    :param perception:
    :param time:
    :return: new classifier or None
    """
    diff = cl.mark.get_differences(perception)

    if diff is None:
        cl.increase_quality()
        return None

    no_spec = cl.specified_unchanging_attributes
    no_spec_new = diff.specificity
    child = cl.copy_from(cl, time)

    if no_spec >= cl.cfg.u_max:
        while no_spec >= cl.cfg.u_max:
            res = cl.generalize_unchanging_condition_attribute(no_spec)
            assert res is True
            no_spec -= 1

        while no_spec + no_spec_new > cl.cfg.u_max:
            if random() < 0.5:
                diff_idx = randint(0, no_spec_new)
                diff.generalize(diff_idx)
                no_spec_new -= 1
            else:
                if cl.generalize_unchanging_condition_attribute(no_spec):
                    no_spec -= 1
    else:
        while no_spec + no_spec_new > cl.cfg.u_max:
            diff_idx = randint(0, no_spec_new)
            diff.generalize(diff_idx)
            no_spec_new -= 1

    child.condition.specialize(new_condition=diff)

    if child.q < 0.5:
        child.q = 0.5

    return child


def unexpected_case(cl: Classifier,
                    previous_perception: Perception,
                    perception: Perception,
                    time: int) -> Classifier:
    """
    Controls the unexpected case of the classifier.

    :param cl:
    :param previous_perception:
    :param perception:
    :param time:
    :return: specialized classifier if generation was possible,
    None otherwise
    """
    cl.decrease_quality()
    cl.set_mark(previous_perception)

    # Return if the effect is not specializable
    if not cl.effect.is_specializable(previous_perception, perception):
        return None

    child = cl.copy_from(cl, time)

    # TODO: p5 maybe also take into consideration cl.E = # (paper)
    child.specialize(previous_perception, perception)

    if child.q < 0.5:
        child.q = 0.5

    return child
