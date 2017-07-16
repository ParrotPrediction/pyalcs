from abc import ABCMeta, abstractmethod
from alcs.helpers.maze_utils import get_all_possible_transitions


class Metric(metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name

    def get(self, *args, **kwargs):
        for kwarg_requirement in self.requirements():
            if kwarg_requirement not in kwargs:
                raise ValueError('Missing keyword argument in {}'
                                 .format(self.__class__))

        return self.name, self.calculate(*args, **kwargs)

    @abstractmethod
    def requirements(self):
        raise NotImplementedError()

    @abstractmethod
    def calculate(self, *args, **kwargs):
        raise NotImplementedError()


class Trial(Metric):
    def requirements(self):
        return ['trial']

    def calculate(self, *args, **kwargs):
        trial_no = kwargs.get('trial')
        return trial_no


class Experiment(Metric):
    def requirements(self):
        return ['experiment']

    def calculate(self, *args, **kwargs):
        experiment_id = kwargs.get('experiment')
        return experiment_id


class StepsInTrial(Metric):
    def requirements(self):
        return ['steps']

    def calculate(self, *args, **kwargs):
        current_step = kwargs.get('steps')
        return current_step


class ClassifierPopulationSize(Metric):
    def requirements(self):
        return ['classifiers']

    def calculate(self, *args, **kwargs):
        classifiers = kwargs.get('classifiers')
        return sum(cl.num for cl in classifiers)


class ReliableClassifierPopulationSize(Metric):
    def requirements(self):
        return ['classifiers']

    def calculate(self, *args, **kwargs):
        classifiers = kwargs.get('classifiers')
        reliable = [cls for cls in classifiers if cls.is_reliable()]
        return sum(cl.num for cl in reliable)


class AveragedFitnessScore(Metric):
    def requirements(self):
        return ['classifiers']

    def calculate(self, *args, **kwargs):
        classifiers = kwargs.get('classifiers')

        sum_fitness = sum(cl.fitness for cl in classifiers)
        total_classifiers = sum(cl.num for cl in classifiers)

        return sum_fitness / total_classifiers


class AverageSpecificity(Metric):
    def requirements(self):
        return ['classifiers']

    def calculate(self, *args, **kwargs):
        classifiers = kwargs.get('classifiers')

        sum_specificity = sum(cl.specificity for cl in classifiers)
        total_classifiers = sum(cl.num for cl in classifiers)

        return sum_specificity / total_classifiers


class AchievedKnowledge(Metric):
    def requirements(self):
        return ['maze', 'classifiers']

    def calculate(self, *args, **kwargs):
        maze = kwargs.get('maze')
        classifiers = kwargs.get('classifiers')

        transitions = get_all_possible_transitions(maze)

        # Take into consideration only reliable classifiers
        reliable_classifiers = [c for c in classifiers if c.is_reliable()]

        # Count how many transitions are anticipated correctly
        nr_correct = 0

        # For all possible destinations from each path cell
        for start, action, end in transitions:

            p0 = maze.get_animat_perception(*start)
            p1 = maze.get_animat_perception(*end)

            if any([True for cl in reliable_classifiers
                    if cl.predicts_successfully(p0, action, p1)]):
                nr_correct += 1

        return nr_correct / len(transitions) * 100.0
