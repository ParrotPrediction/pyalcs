import lcs.agents.acs2 as acs2


class Configuration(acs2.Configuration):
    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

        # ER replay memory buffer size
        self.er_buffer_size = kwargs.get('er_buffer_size', 10000)

        # ER replay memory min samples
        self.er_min_samples = kwargs.get('er_min_samples', 100)

        # ER replay memory samples number
        self.er_samples_number = kwargs.get('er_samples_number', 10)

        # HER new goals to generate number
        self.her_goals_number = kwargs.get('her_goals_number', 3)


    def __str__(self) -> str:
        return str(vars(self))
