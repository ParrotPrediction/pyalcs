
class ClassifiersList(object):
    """
    Represents overall population, match/action sets
    """

    def __init__(self):
        self.list = []

    def form_match_set(self, situation: list):
        match_set = []

        for cls in self.list:
            if cls.condition.does_match(situation):
                match_set.append(cls)

        return match_set

    def get_maximum_fitness(self) -> float:
        """
        Returns the maximum fitness value amongst those classifiers
        that anticipated a change in environment.

        :return: fitness value
        """

        anticipated_change_cls = [cl for cl in self.list if cl.does_anticipate_change()]

        if len(anticipated_change_cls) > 0:
            best_cl = max(anticipated_change_cls, key=lambda cl: cl.fitness)
            return best_cl.fitness

        return 0.0

    def apply_alp(self, previous_situation, action, situation, time, population, match_set) -> None:
        pass

    def apply_reinforcement_learning(self, reward, maximum_fitness) -> None:
        pass

    def apply_ga(self, time, population, match_set, situation) -> None:
        pass


