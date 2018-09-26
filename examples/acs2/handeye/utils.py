def calculate_performance(hand_eye, population):
    """
    Analyzes all possible transition in maze environment and checks if there
    is a reliable classifier for it.
    :param hand_eye: maze object
    :param population: list of classifiers
    :return: percentage of knowledge
    """
    transitions = hand_eye.env.get_all_possible_transitions()

    # Take into consideration only reliable classifiers
    reliable_classifiers = [c for c in population if c.is_reliable()]

    # Count how many transitions are anticipated correctly
    nr_correct = 0
    nr_with_block = 0
    nr_correct_with_block = 0

    # note: knowledge with block only works if env has note_in_hand set to True

    # For all possible destinations from each path cell
    for start, action, end in transitions:

        p0 = start
        p1 = end

        if p0[-1] == '2' or p1[-1] == '2':
            nr_with_block += 1

        if any([True for cl in reliable_classifiers
                if cl.predicts_successfully(p0, action, p1)]):
            nr_correct += 1
            if p0[-1] == '2' or p1[-1] == '2':
                nr_correct_with_block += 1

    return {
        'knowledge': nr_correct / len(transitions) * 100.0,
        'with_block': nr_correct_with_block / nr_with_block * 100.0,
        'no_block':
            (nr_correct - nr_correct_with_block) /
            (len(transitions) - nr_with_block) * 100.0
    }