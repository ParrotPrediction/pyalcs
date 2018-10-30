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


def get_quality_classifiers_list(classifiers: ClassifiersList,
                                 quality: float) -> ClassifiersList:
    """
    Constructs classifier list out of a list with q > quality.

    Parameters
    ----------
    classifiers
    quality

    Returns
    -------

    """
    # TODO: refactor - list comprehension
    listp = ClassifiersList()

    for item in classifiers:
        if item.q > quality:
            listp.append(item)

    return listp


def search_goal_sequence(classifiers: ClassifiersList,
                         start_state: Perception,
                         goal: Perception,
                         theta_r: int) -> list:
    """
    Searches a path from start to goal using a bidirectional method in the
    environmental model (i.e. the list of reliable classifiers).

    Parameters
    ----------
    classifiers: ClassifiersList
    start_state: Perception
    goal: Perception
    theta_r: int
        quality theta_r

    Returns
    -------
    list
        sequence of actions
    """
    reliable_classifiers = \
        get_quality_classifiers_list(classifiers,
                                     quality=theta_r)

    return GoalSequenceSearcher().search_goal_sequence(
        reliable_classifiers, start_state, goal)
