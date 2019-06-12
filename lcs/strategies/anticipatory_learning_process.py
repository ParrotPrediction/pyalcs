from lcs.agents.racs.components.merging import merge
from lcs.strategies.subsumption import does_subsume
import logging

logger = logging.getLogger(__name__)


def add_classifier(child, population, new_list, theta_exp: int) -> None:
    """
    Looks for subsuming / similar classifiers in the population of classifiers
    and those created in the current ALP run (`new_list`).

    If a similar classifier was found it's quality is increased,
    otherwise `child_cl` is added to `new_list`.

    Parameters
    ----------
    child:
        New classifier to examine
    population:
        list of classifiers
    new_list:
        A list of newly created classifiers in this ALP run
    theta_exp: int
        experience threshold for subsumption
    """
    # TODO: p0: write tests
    old_cl = None

    # Look if there is a classifier that subsumes the insertion candidate
    for cl in population:
        if does_subsume(cl, child, theta_exp):
            if old_cl is None or cl.is_more_general(old_cl):
                old_cl = cl

    # Check if any similar classifier was in this ALP run
    if old_cl is None:
        for cl in new_list:
            if cl == child:
                old_cl = cl

    # Check if there is similar classifier already
    if old_cl is None:
        for cl in population:
            if cl == child:
                old_cl = cl

    if old_cl is None:
        to_add = [child]
        # try to merge classifiers
        if child.cfg.do_merging:
            to_add += merge(child, new_list)

        # add all classifiers
        for cl in to_add:
            new_list.append(cl)
    else:
        old_cl.increase_quality()
