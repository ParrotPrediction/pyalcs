from typing import Optional

from lcs import Perception
from lcs.agents.racs import Classifier


def expected_case(cl: Classifier,
                  perception: Perception,
                  time: int) -> Optional[Classifier]:
    # TODO: implement
    raise NotImplementedError()


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
