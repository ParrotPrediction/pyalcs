from typing import Callable


def check_types(oktypes, o):
    if not isinstance(o, oktypes):
        raise TypeError("Wrong element type", o)


def parse_state(raw_state, perception_mapper_fcn: Callable=None):
    """
    Sometimes the environment state returned by the OpenAI
    environment does not suit to the classifier representation
    of data used by ACS2. If a mapping function is defined in
    configuration - use it.

    Parameters
    ----------
    raw_state
        state obtained from OpenAI gym
    perception_mapper_fcn: Callable
        function mapping state
    Returns
    -------
    Perception
        state suitable for an agent (list)
    """
    if perception_mapper_fcn:
        return perception_mapper_fcn(raw_state)

    return raw_state


def parse_action(action_idx: int, action_mapper_fcn: Callable=None) -> int:
    """
    Sometimes the step function from OpenAI Gym takes different
    representation of actions than sequential range of integers.
    There is a possiblity to provide custom mapping function for
    suitable action values.

    Parameters
    ----------
    action_idx: int
        action id, used in PyALCS
    action_mapper_fcn: Callable
        function mapping action

    Returns
    -------
    int
        mapped action id used natively in the environment
    """
    if action_mapper_fcn:
        return action_mapper_fcn(action_idx)

    return action_idx
