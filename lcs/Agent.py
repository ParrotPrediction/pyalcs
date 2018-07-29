class Agent:
    def explore(self, env, trials):
        raise NotImplementedError()

    def exploit(self, env, trials):
        raise NotImplementedError()
