from typing import Optional
import random

from lcs import Perception
from lcs.agents.racs import Classifier


def expected_case(cl: Classifier,
                  perception: Perception,
                  time: int) -> Optional[Classifier]:

    diff = cl.mark.get_differences(perception)

    if diff.specificity == 0:
        cl.increase_quality()
        return None

    no_spec = len(cl.specified_unchanging_attributes)
    no_spec_new = diff.specificity
    child = cl.copy_from(cl, time)

    if no_spec >= cl.cfg.u_max:
        while no_spec >= cl.cfg.u_max:
            res = cl.generalize_unchanging_condition_attribute()
            assert res is True
            no_spec -= 1

        while no_spec + no_spec_new > cl.cfg.u_max:
            if random.random() < 0.5:
                diff.generalize_specific_attribute_randomly()
                no_spec_new -= 1
            else:
                if cl.generalize_unchanging_condition_attribute():
                    no_spec -= 1
    else:
        while no_spec + no_spec_new > cl.cfg.u_max:
            diff.generalize_specific_attribute_randomly()
            no_spec_new -= 1

    child.condition.specialize_with_condition(diff)

    if child.q < 0.5:
        child.q = 0.5

    return child


def unexpected_case(cl: Classifier,
                    previous_perception: Perception,
                    perception: Perception,
                    time: int) -> Optional[Classifier]:
    """
    The classifier does not anticipate the resulting state correctly.
    In this case the classifier is marked by the `previous_perception`
    and it's quality is decreased.

    If it is possible to specialize an offspring (change pass-through
    symbols to correct values then new classifier is returned.

    Parameters
    ----------
    cl: Classifier
        Classifier object
    previous_perception:
        previous situation
    perception: Perception
        current situation
    time:
        current epoch

    Returns
    -------
    Optional[Classifier]
        If possible to specialize parent, None otherwise
    """
    cl.decrease_quality()
    cl.set_mark(previous_perception)

    if not cl.effect.is_specializable(previous_perception, perception):
        return None

    # TODO: p5 maybe also take into consideration cl.E = # (paper)
    child = cl.copy_from(cl, time)
    child.specialize(previous_perception, perception)

    if child.q < .5:
        child.q = .5

    return child
