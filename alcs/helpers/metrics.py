from abc import ABCMeta, abstractmethod
from collections import defaultdict

from alcs.agent.acs2.ACS2Utils import does_anticipate_correctly
from alcs.environment.maze import Maze


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


class ActualStep(Metric):
    def requirements(self):
        return ['step']

    def calculate(self, *args, **kwargs):
        current_step = kwargs.get('step')
        return current_step


class SuccessfulTrial(Metric):
    def requirements(self):
        return ['was_successful']

    def calculate(self, *args, **kwargs):
        success = kwargs.get('was_successful')
        return success


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

        sum_fitness = sum(cl.fitness() for cl in classifiers)
        total_classifiers = len(classifiers)

        return sum_fitness / total_classifiers


class AveragedConditionSpecificity(Metric):
    def requirements(self):
        return ['classifiers']

    def calculate(self, *args, **kwargs):
        classifiers = kwargs.get('classifiers')

        sum_specificity = sum(cl.condition.get_specificity()
                              for cl in classifiers)
        total_classifiers = len(classifiers)

        return sum_specificity / total_classifiers


class AchievedKnowledge(Metric):
    def requirements(self):
        return ['maze', 'classifiers']

    def calculate(self, *args, **kwargs):
        env = kwargs.get('maze')
        classifiers = kwargs.get('classifiers')

        reliable_classifiers = [c for c in classifiers if c.is_reliable()]

        # Filter classifiers for each possible action
        n_classifiers = [c for c in reliable_classifiers if c.action == 0]
        ne_classifiers = [c for c in reliable_classifiers if c.action == 1]
        e_classifiers = [c for c in reliable_classifiers if c.action == 2]
        se_classifiers = [c for c in reliable_classifiers if c.action == 3]
        s_classifiers = [c for c in reliable_classifiers if c.action == 4]
        sw_classifiers = [c for c in reliable_classifiers if c.action == 5]
        w_classifiers = [c for c in reliable_classifiers if c.action == 6]
        nw_classifiers = [c for c in reliable_classifiers if c.action == 7]

        possible_moves = 0
        reliable_moves = 0

        path_possible_destinations = defaultdict(list)

        # For all possible destinations from each path cell
        for pos_x, pos_y in env.get_possible_agent_insertion_coordinates():
            neighbour_cells = env.get_possible_neighbour_cords(pos_x, pos_y)
            allowed_cells = [c for c in neighbour_cells
                             if env.is_path(*c) or env.is_reward(*c)]

            path_possible_destinations[(pos_x, pos_y)] = allowed_cells

        # Now, for each possible transition check if there is a classifier
        for start, destinations in path_possible_destinations.items():
            for destination in destinations:
                possible_moves += 1

                previous_perception = env.get_animat_perception(start[0],
                                                                start[1])
                perception = env.get_animat_perception(destination[0],
                                                       destination[1])

                # North transition
                if Maze.moved_north(start, destination):
                    if any(does_anticipate_correctly(cl, perception,
                                                     previous_perception)
                           for cl in n_classifiers):
                        reliable_moves += 1

                # North-East transition
                if Maze.moved_north(start, destination) \
                        and Maze.moved_east(start, destination):
                    if any(does_anticipate_correctly(cl, perception,
                                                     previous_perception)
                           for cl in ne_classifiers):
                        reliable_moves += 1

                # East transition
                if Maze.moved_east(start, destination):
                    if any(does_anticipate_correctly(cl, perception,
                                                     previous_perception)
                           for cl in e_classifiers):
                        reliable_moves += 1

                # South-East transition
                if Maze.moved_south(start, destination) \
                        and Maze.moved_east(start, destination):
                    if any(does_anticipate_correctly(cl, perception,
                                                     previous_perception)
                           for cl in se_classifiers):
                        reliable_moves += 1

                # South transition
                if Maze.moved_south(start, destination):
                    if any(does_anticipate_correctly(cl, perception,
                                                     previous_perception)
                           for cl in s_classifiers):
                        reliable_moves += 1

                # South-West transition
                if Maze.moved_south(start, destination) \
                        and Maze.moved_west(start, destination):
                    if any(does_anticipate_correctly(cl, perception,
                                                     previous_perception)
                           for cl in sw_classifiers):
                        reliable_moves += 1

                # West transition
                if Maze.moved_west(start, destination):
                    if any(does_anticipate_correctly(cl, perception,
                                                     previous_perception)
                           for cl in w_classifiers):
                        reliable_moves += 1

                # North-West transition
                if Maze.moved_north(start, destination) \
                        and Maze.moved_west(start, destination):
                    if any(does_anticipate_correctly(cl, perception,
                                                     previous_perception)
                           for cl in nw_classifiers):
                        reliable_moves += 1

        return reliable_moves / possible_moves
