from lcs import Perception


def state_goal_concat(state: Perception, goal: Perception) -> Perception:
    return Perception(tuple(state) + tuple(goal))
