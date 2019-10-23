def update_classifier(cl, step_reward: int, max_fitness: float,
                      beta: float, gamma: float):
    """
    Applies Reinforcement Learning according to
    current reinforcement `reward` and back-propagated reinforcement
    `maximum_fitness`.

    Classifier parameters are updated.

    Parameters
    ----------
    cl:
        classifier with `r` and `ir` properties
    step_reward: int
        current reward obtained from the environment after executing step
    max_fitness: float
        maximum fitness - back-propagated reinforcement. Maximum fitness
        from the match set
    beta: float
    gamma: float
    """

    _discounted_reward = step_reward + gamma * max_fitness

    # Update classifier properties
    cl.r += beta * (_discounted_reward - cl.r)
    cl.ir += beta * (step_reward - cl.ir)
