import lcs.agents.acs2 as acs2


class Configuration(acs2.Configuration):
    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

        # ER replay memory buffer size
        self.er_buffer_size = kwargs.get('er_buffer_size', 10000)

        # ER replay memory min samples
        self.er_min_samples = kwargs.get('er_min_samples', 1000)

        # ER replay memory samples number
        self.er_samples_number = kwargs.get('er_samples_number', 3)

        # HER new goals to generate strategy
        self.her_strategy = kwargs.get('her_strategy', 'future')

        # HER new goals to generate number
        self.her_new_goals_number = kwargs.get('her_new_goals_number', 3)

        # HER penalty reward
        self.her_penalty_reward = kwargs.get('her_penalty_reward', -100)

    def __str__(self) -> str:
        return str(vars(self))
