from typing import List

from lcs import Perception
from lcs.agents.acs2 import ClassifiersList
from lcs.strategies.action_planning.goal_sequence_searcher \
    import GoalSequenceSearcher


def exists_classifier(classifiers: ClassifiersList,
                      p0: Perception,
                      action: int,
                      p1: Perception,
                      quality: float) -> bool:
    """

    Parameters
    ----------
    classifiers
    p0: Perception
        previous situation
    action: int
    p1: Perception
        situation
    quality: float

    Returns
    -------
    bool
        True if there is a classifier in this list with a quality
        higher than 'quality' that matches previous_situation,
        specifies action, and predicts situation.
        False otherwise.
    """
    for cl in classifiers:
        if cl.q > quality and cl.does_match(p0) \
            and cl.action == action \
            and cl.does_anticipate_correctly(p0,
                                             p1):
            return True
    return False


def search_goal_sequence(classifiers: ClassifiersList,
                         p0: Perception,
                         p1: Perception) -> List:
    """
    Searches a path from start to goal using a bidirectional method in the
    environmental model (i.e. the list of reliable classifiers).

    Parameters
    ----------
    classifiers: ClassifiersList
    p0: Perception
        start state
    p1: Perception
        destination state

    Returns
    -------
    list
        sequence of actions
    """
    reliable = [cl for cl in classifiers if cl.is_reliable()]
    gs = GoalSequenceSearcher()

    return gs.search_goal_sequence(ClassifiersList(*reliable), p0, p1)
