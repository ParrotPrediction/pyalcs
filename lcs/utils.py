def check_types(oktypes, o):
    if not isinstance(o, oktypes):
        raise TypeError("Wrong element type", o)


def parse_state(raw_state, perception_mapper_fcn=None):
    """
    Sometimes the environment state returned by the OpenAI
    environment does not suit to the classifier representation
    of data used by ACS2. If a mapping function is defined in
    configuration - use it.

    Parameters
    ----------
    raw_state
        state obtained from OpenAI gym
    perception_mapper_fcn
        function mapping state
    Returns
    -------
        state suitable for an agent (list)
    """
    if perception_mapper_fcn:
        return perception_mapper_fcn(raw_state)

    return raw_state
