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
