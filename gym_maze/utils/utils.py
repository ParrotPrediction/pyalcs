import networkx as nx

from gym_maze import find_action_by_direction
from gym_maze.maze import Maze


def get_all_possible_transitions(maze):
    """
    Returns all possible transitions within the maze.
    [POINT]->[ACTION]->[POINT]
    This information is used to calculate the agent's knowledge
    :param maze: an instance of the maze
    :return: 
    """
    transitions = []

    g = _create_graph(maze)

    path_nodes = (node for node, data
                  in g.nodes(data=True) if data['type'] == 'path')

    for node in path_nodes:
        for neighbour in nx.all_neighbors(g, node):
            direction = Maze.distinguish_direction(node, neighbour)
            action = find_action_by_direction(direction)

            transitions.append((node, action, neighbour))

    return transitions


def _create_graph(env):
    maze = env.maze

    # Create uni-directed graph
    g = nx.Graph()

    # Add nodes
    for x in range(0, maze.max_x):
        for y in range(0, maze.max_y):
            if maze.is_path(x, y):
                g.add_node((x, y), type='path')
            if maze.is_reward(x, y):
                g.add_node((x, y), type='reward')

    # Add edges
    path_nodes = [cords for cords, attribs
                  in g.nodes(data=True) if attribs['type'] == 'path']

    for n in path_nodes:
        neighbour_cells = Maze.get_possible_neighbour_cords(*n)
        allowed_cells = [c for c in neighbour_cells
                         if maze.is_path(*c) or maze.is_reward(*c)]
        edges = [(n, dest) for dest in allowed_cells]

        g.add_edges_from(edges)

    return g