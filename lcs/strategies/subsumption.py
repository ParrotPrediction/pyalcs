def does_subsume(cl, other_cl, theta_exp: int) -> bool:
    """
    Returns if a classifier `cl` subsumes `other_cl` classifier

    Parameters
    ----------
    cl:
        subsumer classifier
    other:
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
        cl.condition.does_match_condition(other_cl.condition) and \
            cl.action == other_cl.action and \
            cl.effect == other_cl.effect:
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
