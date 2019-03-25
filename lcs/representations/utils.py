import statistics

from lcs.representations import UBR
from lcs.representations.RealValueEncoder import RealValueEncoder


def cover_ratio(ps, encoder: RealValueEncoder) -> float:
    """
    Calculates the perception space covered by condition attribute.
    An arithmetic average is taken over all condition attributes

    Parameters:
    ps: Perception String, either Condition or Effect from rACS module

    Returns
    -------
    float
        A value between (0, 1], where
        0.0 means that condition is nothing is covered
        1.0 means that condition is maximally general
    """
    maximum_span = encoder.range[1] + 1
    return statistics.mean(r.bound_span / maximum_span for r in ps)


def add_from_both_sides(u: UBR, val: int, lb: int, ub: int):
    """
    Adds the same value to UBR interval (genotype) from both sides.
    Each size is trimmed to `lb` or `ub` if needed.

    Also, the attributes of the interval are sorted during the process.
    This might break the genome of the UBR, when sometimes lower value might
    be placed as a second parameter.

    Parameters
    ----------
    u: UBR
        interval
    val: encoded value to be added
    lb: int
        minimum possible value (lower bound of the encoder)
    ub: int
        maximum possible value (upper bound of the encoder)
    """
    old_lb, old_ub = u.lower_bound, u.upper_bound

    u.x1 = old_lb - val if (old_lb - val) > lb else lb
    u.x2 = old_ub + val if (old_ub + val) < ub else ub
