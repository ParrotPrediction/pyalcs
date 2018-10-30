from typing import Dict


def handeye_metrics(population, environment) -> Dict:
    """
    Analyzes all possible transition in maze environment and checks if there
    is a reliable classifier for it.

    Note: knowledge with/without block only works if env has note_in_hand
    set to True

    Parameters
    ----------
    population
        list of classifiers
    environment
        handeye environment

    Returns
    -------
    Dict
        knowledge - percentage of transitions we are able to anticipate
            correctly (max 100)

        with_block - percentage of all transitions involving block -
            gripping, realising or moving block - we are able to anticipate
            correctly (max 100)

        no_block - percentage of all transitions not involving block -
            moving the gripper without block - we are able to anticipate
            correctly (max 100)
    """
    transitions = environment.env.get_all_possible_transitions()

    # Take into consideration only reliable classifiers
    reliable_classifiers = [c for c in population if c.is_reliable()]

    # Count how many transitions are anticipated correctly
    nr_correct = 0
    nr_with_block = 0
    nr_correct_with_block = 0

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
