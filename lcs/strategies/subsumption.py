from typing import List


def find_subsumers(cl, population, theta_exp: int) -> List:
    """
    Looks for subsumers of `cl` inside `population`.

    Parameters
    ----------
    cl:
        classifier
    population:
        population of classifiers
    theta_exp: int
        subsumption experience threshold

    Returns
    -------
    List
        list of subsumers (classifiers) sorted by specificity (most general
        are first)
    """
    subsumers = [sub for sub in population if does_subsume(sub, cl, theta_exp)]
    return sorted(subsumers, key=lambda cl: cl.condition.specificity)


def does_subsume(cl, other_cl, theta_exp: int) -> bool:
    """
    Returns if a classifier `cl` subsumes `other_cl` classifier

    Parameters
    ----------
    cl:
        subsumer classifier
    other_cl:
        other classifier
    theta_exp: int
        experience threshold

    Returns
    -------
    bool
        True if `other_cl` classifier is subsumed, False otherwise
    """
    if is_subsumer(cl, theta_exp) and \
        cl.is_more_general(other_cl) and \
        cl.condition.subsumes(other_cl.condition) and \
            cl.action == other_cl.action and \
            cl.effect.subsumes(other_cl.effect):
        return True

    return False


def is_subsumer(cl, theta_exp: int) -> bool:
    """
    Determines whether the classifier satisfies the subsumer criteria.

    Parameters
    ----------
    cl:
        classifier
    theta_exp: int
        Experience threshold to be considered as experienced

    Returns
    -------
    bool
        True is classifier can be considered as subsumer,
        False otherwise
    """
    if cl.exp > theta_exp:
        if cl.is_reliable():
            if not cl.is_marked():
                return True

    return False
