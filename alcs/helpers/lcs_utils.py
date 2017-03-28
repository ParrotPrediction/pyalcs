from itertools import chain


def get_dynamic_exploration_probability(step, steps) -> float:
    """
    Calculates exploration probability accordingly to the current step.
    The agent is more willing to explore at the beginning, and to
    exploit at the end
    :param step: current step
    :param steps: max steps
    :return: exploration rate for a given step
    """
    unit = step / float(steps)
    return 1 - unit


def unwind_micro_classifiers(macro_classifiers: list) -> list:
    """
    Transforms a list of macro=classifiers into micro-classifiers (accordingly
    to num parameter)

    :param macro_classifiers: a list of macro-classifiers
    :return: a list of micro-classifiers
    """
    micros_packed = [cl.get_micro_classifiers() for cl in macro_classifiers]
    return list(chain(*micros_packed))
