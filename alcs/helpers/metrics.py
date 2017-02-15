from collections import defaultdict

from alcs.environment.maze import Maze
from alcs.agent.acs2.ACS2Utils import does_anticipate_correctly


def calculate_achieved_knowledge(env: Maze,
                                 classifiers: list) -> float:
    """
    Calculates achieved knowledge for maze environment.

    :param env: given maze environment
    :param classifiers: population of classifiers (list)
    :return: percentage of learned knowledge [0,1]
    """

    reliable_classifiers = [c for c in classifiers if c.q > 0.9]

    # Filter classifiers for each possible action
    north_classifiers = [c for c in reliable_classifiers if c.action == 1]
    south_classifiers = [c for c in reliable_classifiers if c.action == 3]
    west_classifiers = [c for c in reliable_classifiers if c.action == 0]
    east_classifiers = [c for c in reliable_classifiers if c.action == 2]

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

            previous_perception = env.get_animat_perception(start[0], start[1])
            perception = env.get_animat_perception(destination[0],
                                                   destination[1])

            # North transition (y+1)
            if destination[1] + 1 == start[1]:
                if any(does_anticipate_correctly(cl, perception,
                                                 previous_perception)
                       for cl in north_classifiers):
                    reliable_moves += 1

            # West transition (x+1)
            if destination[0] + 1 == start[0]:
                if any(does_anticipate_correctly(cl, perception,
                                                 previous_perception)
                       for cl in west_classifiers):
                    reliable_moves += 1

            # East transition (x-1)
            if destination[0] - 1 == start[0]:
                if any(does_anticipate_correctly(cl, perception,
                                                 previous_perception)
                       for cl in east_classifiers):
                    reliable_moves += 1

            # South transition (y-1)
            if destination[1] - 1 == start[1]:
                if any(does_anticipate_correctly(cl, perception,
                                                 previous_perception)
                       for cl in south_classifiers):
                    reliable_moves += 1

    return reliable_moves / possible_moves
