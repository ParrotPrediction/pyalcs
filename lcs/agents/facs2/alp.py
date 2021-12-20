from random import random
from typing import Optional

from lcs import Perception
from lcs.agents.facs2 import Classifier, Configuration


def cover(p0: Perception,
          action: int,
          p1: Perception,
          time: int,
          cfg: Configuration) -> Classifier:

    new_cl = Classifier(action=action, experience=0, reward=0, cfg=cfg)
    new_cl.tga = time
    new_cl.talp = time

    new_cl.specialize(p0, p1)

    return new_cl


def expected_case(cl: Classifier,
                  p0: Perception,
                  time: int) -> Optional[Classifier]:


    diff = cl.mark.get_differences(p0)

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
            if random() < 0.5:
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
                    p0: Perception,
                    p1: Perception,
                    time: int) -> Optional[Classifier]:

    cl.decrease_quality()
    cl.set_mark(p0)

    if not cl.effect.is_specializable(p0, p1):
        return None

    child = cl.copy_from(cl, time)

    child.specialize(p0, p1, leave_specialized=True)

    if child.q < 0.5:
        child.q = 0.5

    return child
