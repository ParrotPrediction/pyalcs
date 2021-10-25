import lcs.agents.acs2 as acs2


class Configuration(acs2.Configuration):
    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

        self.zeta: float = kwargs.get('zeta', 0.001)
        self.rho_update_version: str = kwargs.get('rho_update_version', '1')

    def __str__(self) -> str:
        return str(vars(self))
