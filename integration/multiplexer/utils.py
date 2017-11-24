from gym_multiplexer.utils import get_correct_answer


def calculate_performance(env, population, ctrl_bits=None):
    p1 = env.render()  # state after executing action
    p0 = p1[:-1] + '0'  # initial state
    correct_answer = get_correct_answer(p0, ctrl_bits)  # true action

    reliable_classifiers = [c for c in population if c.is_reliable()]

    cl_exists = any([1 for cl in reliable_classifiers if
                     cl.predicts_successfully(p0, correct_answer, p1)])

    return {
        'was_correct': cl_exists
    }
