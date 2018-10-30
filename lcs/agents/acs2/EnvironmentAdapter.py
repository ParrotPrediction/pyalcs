class EnvironmentAdapter:
    """
    Sometimes the observation returned by the OpenAI Gym environment
    does not suit the observation representation as used by the ACS2.
    In that case there's a need for converter functions that map the ACS2
    representation to the environment representation and vice versa.

    The same goes for action representation.

    This class realizes this conversion.

    This implementation works for environments which provide ACS2-compatible
    state and action representations (no conversion needed).

    Subclass this class in pyALCS integration to provide an adapter
    for a specific environment.
    """

    @staticmethod
    def env_action_to_acs(env_action):
        """
        Converts environment representation of an action to ACS2
        representation.
        """
        return env_action

    @staticmethod
    def env_state_to_acs(env_state):
        """
        Converts environment representation of a state to ACS2
        representation.
        """
        return env_state

    @staticmethod
    def acs_action_to_env(acs_action):
        """
        Converts ACS2 representation of an action to environment
        representation.
        """
        return acs_action

    @staticmethod
    def acs_state_to_env(acs_state):
        """
        Converts ACS2 representation of a state to environment
        representation.
        """
        return acs_state
