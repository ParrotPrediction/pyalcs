class EnvironmentAdapter:
    """
    Sometimes the observation returned by the OpenAI Gym environment
    does not suit the observation representation as used by the LCS.
    In that case there's a need for converter functions that map the LCS
    representation  to the environment representation and vice versa.

    The same goes for action representation.

    This class realizes this conversion.

    This implementation works for environments which provide LCS-compatible
    state and action representations (no conversion needed).

    Subclass this class in PyALCS integration to provide an adapter
    for a specific environment.
    """

    @classmethod
    def to_lcs_action(cls, env_action):
        """
        Converts environment representation of an action to LCS
        representation.
        """
        return env_action

    @classmethod
    def to_genotype(cls, phenotype):
        """
        Converts environment representation of a state to LCS
        representation.
        """
        return phenotype

    @classmethod
    def to_env_action(cls, lcs_action):
        """
        Converts LCS representation of an action to environment
        representation.
        """
        return lcs_action

    @classmethod
    def to_phenotype(cls, genotype):
        """
        Converts LCS representation of a state to environment
        representation.
        """
        return genotype
