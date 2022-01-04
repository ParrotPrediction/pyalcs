import lcs.agents.acs as acs


class Configuration(acs.Configuration):
    def __init__(self, **kwargs):

        super(Configuration, self).__init__(**kwargs)

        self.gamma: float = kwargs.get('gamma', 0.95)
        self.do_ga: bool = kwargs.get('do_ga', False)
        self.initial_q: float = kwargs.get('initial_q', 0.5)
        self.biased_exploration_prob: float = kwargs.get(
            'biased_exploration_prob', 0.05)
        self.theta_ga: int = kwargs.get('theta_ga', 100)
        self.mu: float = kwargs.get('mu', 0.3)
        self.chi: float = kwargs.get('chi', 0.8)

    def __str__(self) -> str:
        return str(vars(self))

