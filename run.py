from environment import Environment, Maze, MazeAction
from agent.acs2.ACS2Utils import *

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=logging.DEBUG)

if __name__ == '__main__':
    GENERATIONS = 5

    env = Maze('mazes/m1.maze')

    time = 0
    classifiers = []
    match_set = []
    action_set = []

    previous_action_set = None
    previous_perception = None

    for _ in range(GENERATIONS):
        logger.info('\n\nGeneration [%d]', time)

        if time == 0 or env.animat_found_reward:
            env.reset_animat_state()
            env.insert_animat()

        if time == 0 or len(classifiers) == 0:
            classifiers = generate_initial_classifiers()

        perception = list(env.get_animat_perception())
        match_set = generate_match_set(classifiers, perception)

        # Here do some stuff on previous action set
        # ALP.apply(classifiers, None, time, action_set, [], [])

        action = choose_action(match_set)
        action_set = generate_action_set(match_set, action)

        time += 1

    logger.info('Finished')
