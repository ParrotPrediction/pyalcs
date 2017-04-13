from __future__ import division

from alcs.environment.maze import Maze


def calculate_optimal_path_length(maze: Maze) -> float:
    """
    Returns a optimal number of steps to finding a reward.

    This function calculates the shortest path from each possible point in
    the maze and returns an averaged result.
    """
    import networkx as nx

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
    reward_node = [cords for cords, attribs
                   in g.nodes(data=True) if attribs['type'] == 'reward'][0]

    for n in path_nodes:
        neighbour_cells = maze.get_possible_neighbour_cords(*n)
        allowed_cells = [c for c in neighbour_cells
                         if maze.is_path(*c) or maze.is_reward(*c)]
        edges = [(n, dest) for dest in allowed_cells]

        g.add_edges_from(edges)

    # Calculate shortest paths from each node
    distances = {pn: nx.shortest_path_length(g, pn, reward_node)
                 for pn in path_nodes}

    # Calculate average shortest path
    return sum(distances.values()) / len(distances)
