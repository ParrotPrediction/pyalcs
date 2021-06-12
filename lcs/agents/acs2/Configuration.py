import lcs.agents.acs as acs
from lcs.strategies.action_selection import EpsilonGreedy


class Configuration(acs.Configuration):
    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

        # RL discount factor
        self.gamma: float = kwargs.get('gamma', 0.95)

        # use of Probability-Enhanced-Effects (PEE)
        self.do_pee: bool = kwargs.get('do_pee', False)

        # use of genetic generalization algorithm
        self.do_ga: bool = kwargs.get('do_ga', False)

        self.theta_ga: int = kwargs.get('theta_ga', 100)

        # probability of mutating single attribute
        self.mu: float = kwargs.get('mu', 0.3)

        # probability of crossover
        self.chi: float = kwargs.get('chi', 0.8)

        # use of action planning mechanism
        self.do_action_planning: bool = kwargs.get('do_action_planning', False)

        self.action_planning_frequency: int = kwargs.get(
            'action_planning_frequency', 50)

        # initial quality assigned to classifiers
        self.initial_q: float = kwargs.get('initial_q', 0.5)

        # probability of executing biased exploration phases
        self.biased_exploration_prob: float = kwargs.get(
            'biased_exploration_prob', 0.05)

        self.action_selector = kwargs.get('action_selector', EpsilonGreedy)(
            all_actions=self.number_of_possible_actions,
            epsilon=self.epsilon,
            biased_exploration_prob=self.biased_exploration_prob
        )

    def __str__(self) -> str:
        return str(vars(self))
