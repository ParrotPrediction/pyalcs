from typing import Optional

from lcs import Perception
from lcs.agents.racs import Classifier


def unexpected_case(cl: Classifier,
                    previous_perception: Perception,
                    perception: Perception,
                    time: int) -> Optional[Classifier]:
    # TODO: write tests
    cl.decrease_quality()
    cl.set_mark(previous_perception)

    if not cl.effect.is_specializable(previous_perception, perception):
        return None

    # TODO: p5 maybe also take into consideration cl.E = # (paper)
    child = cl.copy_from(cl, time)
    child.specialize(previous_perception, perception)

    if child.q < 0.5:
        child.q = 0.5

    return child
