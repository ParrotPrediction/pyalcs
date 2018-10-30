from gym_multiplexer.utils import get_correct_answer


def reliable_cl_exists(env, population, ctrl_bits=None) -> bool:
    p1 = env.render('ansi')  # state after executing action
    p0 = p1[:-1] + ['0']  # initial state
    correct_answer = get_correct_answer(p0, ctrl_bits)  # true action

    reliable_classifiers = [c for c in population if c.is_reliable()]

    return any([1 for cl in reliable_classifiers if
                cl.predicts_successfully(p0, correct_answer, p1)])
