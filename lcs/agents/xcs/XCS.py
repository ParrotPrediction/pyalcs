from lcs.agents import Agent


class Configuration:
    def __init__(self):
        raise NotImplementedError()


class Classifier:
    def __init__(self):
        raise NotImplementedError()


class ClassifierList:
    def __init__(self):
        raise NotImplementedError()

    def generate_match_set(self):
        raise NotImplementedError()


class XCS(Agent):
    def __init__(self,
                 cfg: Configuration,
                 population: ClassifierList):
        raise NotImplementedError()

    def get_population(self):
        return self.cfg

    def get_cfg(self):
        return self.population
