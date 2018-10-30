def maze_knowledge(population, environment) -> float:
    """
    Analyzes all possible transition in maze environment and checks if there
    is a reliable classifier for it.

    Parameters
    ----------
    population
        list of classifiers
    environment
        maze object

    Returns
    -------
    float
        percentage of knowledge
    """
    transitions = environment.env.get_all_possible_transitions()

    # Take into consideration only reliable classifiers
    reliable_classifiers = [c for c in population if c.is_reliable()]

    # Count how many transitions are anticipated correctly
    nr_correct = 0

    # For all possible destinations from each path cell
    for start, action, end in transitions:

        p0 = environment.env.maze.perception(*start)
        p1 = environment.env.maze.perception(*end)

        if any([True for cl in reliable_classifiers
                if cl.predicts_successfully(p0, action, p1)]):
            nr_correct += 1

    return nr_correct / len(transitions) * 100.0


def print_detailed_knowledge(maze, population):
    transitions = maze.env.get_all_possible_transitions()

    # For all possible destinations from each path cell
    for start, action, end in transitions:

        p0 = maze.env.maze.perception(*start)
        p1 = maze.env.maze.perception(*end)

        print("\n{}-{}-\n{}".format("".join(p0), action, "".join(p1)))
        print(population.form_match_set(p0).form_action_set(action))
