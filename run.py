from environment import Environment, Maze, MazeAction
from agent import ACS2
from agent.acs2 import Constants as const

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    env = Maze('mazes/m1.maze')

    time = 0
    classifiers = []
    match_set = []
    action_set = []

    for _ in range(5):
        logger.info('\n\nGeneration [%d]', time)

        perception = env.get_animat_perception()

        if time == 0 or len(classifiers) == 0:
            classifiers = ACS2.generate_initial_classifiers(const.AGENT_NUMBER_OF_POSSIBLE_ACTIONS)

        match_set = ACS2.generate_match_set(classifiers, perception)

        # Here do some stuff on previous action set

        action = ACS2.choose_action(match_set)
        action_set = ACS2.generate_action_set(match_set, action)

        time += 1

    logger.info('Finished')

