from lcs import Perception
from lcs.agents.acs import Classifier, ClassifiersList


def apply(p0: Perception, p1: Perception, cl: Classifier, population: ClassifiersList):
    if not _perception_changed(p0, p1):
        handle_useless_case(cl)
    elif cl.does_anticipate_correctly(p0, p1):
        handle_expected_case(cl)
    elif not cl.does_anticipate_correctly(p0, p1) and _perception_changed(p0, p1) and cl.can_be_corrected(p0, p1):
        handle_correctable_case(p0, p1, cl, population)
    else:
        handle_not_correctable_case(cl, p0)

    # Remove inadequate classifiers
    for cl in population:
        if cl.is_inadequate() and not cl.is_general():
            population.remove(cl)


def handle_useless_case(cl: Classifier):
    cl.decrease_quality()


def handle_expected_case(cl: Classifier):
    cl.increase_quality()


def handle_correctable_case(p0: Perception, p1: Perception, cl: Classifier, population: ClassifiersList):
    new_cl = Classifier.build_corrected(cl, p0, p1)
    existing = [cl for cl in population if cl == new_cl]

    if len(existing) == 0:
        population.append(new_cl)


def handle_not_correctable_case(cl: Classifier, p0: Perception):
    cl.decrease_quality()
    cl.set_mark(p0)


def _perception_changed(p0: Perception, p1: Perception) -> bool:
    return p0 != p1
