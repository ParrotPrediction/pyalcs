import functools

from lcs import Perception
from lcs.agents.acs2 import Condition


@functools.lru_cache(maxsize=16384)
def matching(c: Condition, p: Perception) -> bool:
    """
    Check if condition match other list such as perception or another
    condition.

    Parameters
    ----------
    c: Condition
    p: Perception

    Returns
    -------
    bool
        True if condition match given list, False otherwise
    """
    for ci, oi in zip(c, p):
        if ci != c.WILDCARD and oi != c.WILDCARD and ci != oi:
            return False

    return True
