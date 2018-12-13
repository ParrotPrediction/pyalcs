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


def detailed_knowledge(maze, population):
    """
    Analyze the population of classifiers to determine what classifiers cover
    all possible condition-action tuples in the maze.

    For every condition-action tuple (i.e., every situation together with
    all allowed actions), print the classifiers that match it.

    :param maze: The maze to analyze
    :param population: The classifier population to analyze
    :return: String describing the population's knowledge as in the original
    C++ implementation
    """
    result = ""
    transitions = maze.env.get_all_possible_transitions()

    # For all possible destinations from each path cell
    for start, action, end in transitions:

        p0 = maze.env.maze.perception(*start)
        p1 = maze.env.maze.perception(*end)

        result += "\n{}-{}-\n{}".format("".join(p0), action, "".join(p1))
        result += "\n"
        result += str(population.form_match_set(p0).form_action_set(action))
        result += "\n"

    return result
