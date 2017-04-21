from alcs.agent import Perception
from alcs.agent.acs3.Classifier import Classifier


class ClassifiersList(list):
    """
    Represents overall population, match/action sets
    """

    def __init__(self, *args):
        list.__init__(self, *args)

    def append(self, item):
        if not isinstance(item, Classifier):
            raise TypeError("Item should be a Classifier object")

        super(ClassifiersList, self).append(item)

    @classmethod
    def form_match_set(cls, population, situation: Perception):
        match_set = cls()

        for cl in population:
            if cl.condition.does_match(situation):
                match_set.append(cl)

        return match_set

    def get_maximum_fitness(self) -> float:
        """
        Returns the maximum fitness value amongst those classifiers
        that anticipated a change in environment.

        :return: fitness value
        """
        anticipated_change_cls = [cl for cl in self if cl.does_anticipate_change()]

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
