from __future__ import division

import networkx as nx
from alcs.environment.maze import Maze, MazeAction


def get_all_possible_transitions(maze: Maze):
    """
    Returns all possible transitions within the maze.
    [POINT]->[ACTION]->[POINT]
    This information is used to calculate the agent's knowledge
    
    :param maze: 
    :return: 
    """
    ma = MazeAction()
    transitions = []

    g = _create_graph(maze)

    path_nodes = (node for node, data
                  in g.nodes_iter(data=True) if data['type'] == 'path')

    for node in path_nodes:
        for neighbour in nx.all_neighbors(g, node):
            direction = _distinguish_direction(node, neighbour)
            action = ma[direction]['value']

            transitions.append((node, action, neighbour))

    return transitions


def calculate_optimal_path_length(maze: Maze) -> float:
    """
    Returns a optimal number of steps to finding a reward.

    This function calculates the shortest path from each possible point in
    the maze and returns an averaged result.
    """
    g = _create_graph(maze)

    path_nodes = [cords for cords, attribs
                  in g.nodes(data=True) if attribs['type'] == 'path']

    reward_node = [cords for cords, attribs
                   in g.nodes(data=True) if attribs['type'] == 'reward'][0]

    # Calculate shortest paths from each node
    distances = {pn: nx.shortest_path_length(g, pn, reward_node)
                 for pn in path_nodes}

    # Calculate average shortest path
    return sum(distances.values()) / len(distances)


def _create_graph(maze: Maze):
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
        neighbour_cells = maze.get_possible_neighbour_cords(*n)
        allowed_cells = [c for c in neighbour_cells
                         if maze.is_path(*c) or maze.is_reward(*c)]
        edges = [(n, dest) for dest in allowed_cells]

        g.add_edges_from(edges)

    return g


def _distinguish_direction(start, end):
    direction = ''

    if Maze.moved_north(start, end):
        direction += 'N'

    if Maze.moved_south(start, end):
        direction += 'S'

    if Maze.moved_west(start, end):
        direction += 'W'

    if Maze.moved_east(start, end):
        direction += 'E'

    return direction
