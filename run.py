from environment import Environment, Maze, MazeAction
from agent import ACS2Utils, ALP
from agent.acs2 import Constants as const

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=logging.DEBUG)

if __name__ == '__main__':
    env = Maze('mazes/m1.maze')

    time = 0
    classifiers = []
    match_set = []
    action_set = []

    previous_action_set = None
    previous_perception = None

    for _ in range(5):
        logger.info('\n\nGeneration [%d]', time)

        if time == 0 or env.animat_found_reward:
            env.reset_animat_state()
            env.insert_animat()

        if time == 0 or len(classifiers) == 0:
            classifiers = ACS2Utils.generate_initial_classifiers()

        perception = env.get_animat_perception()
        match_set = ACS2Utils.generate_match_set(classifiers, perception)

        # Here do some stuff on previous action set
        # ALP.apply(classifiers, None, time, action_set, [], [])

        action = ACS2Utils.choose_action(match_set)
        action_set = ACS2Utils.generate_action_set(match_set, action)

        time += 1

    logger.info('Finished')
