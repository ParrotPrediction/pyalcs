import lcs.agents.acs2 as acs2


class Configuration(acs2.Configuration):
    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

        # ER replay memory buffer size
        self.er_buffer_size = kwargs.get('er_buffer_size', 10000)

        # ER replay memory samples number
        self.er_samples_number = kwargs.get('er_samples_number', 8)

        # HER generating goals strategy
        self.her_strategy = kwargs.get('her_strategy', None)

        # HER new goals to generate number
        self.her_goals_number = kwargs.get('her_goals_number', 3)

        # HER fn reward
        self.her_reward_generator = kwargs.get('her_reward_generator', None)

    def __str__(self) -> str:
        return str(vars(self))
