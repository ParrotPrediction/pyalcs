import sys
import logging
from os.path import abspath, join, dirname

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from acs.agent import acs2
from acs.environment import maze


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=logging.DEBUG)

if __name__ == '__main__':
    GENERATIONS = 5

    env = maze.Maze('mazes/m1.maze')

    time = 0
    classifiers = []
    match_set = []
    action_set = []

    action = None
    reward = None
    previous_action_set = None
    previous_perception = None

    for _ in range(GENERATIONS):
        logger.info('\n\nGeneration [%d]', time)

        # Reset the environment and put the animat randomly
        # inside the maze when we are starting the simulation
        # or when he found the reward (next trial)
        if time == 0 or env.animat_found_reward:
            env.reset_animat_state()
            env.insert_animat()

        # Generate initial (general) classifiers when the simulation
        # Just starts or there is none classifier in the population
        if time == 0 or len(classifiers) == 0:
            classifiers = acs2.generate_initial_classifiers()

        # Get the animat perception
        perception = list(env.get_animat_perception())

        # Select classifiers matching the perception
        match_set = acs2.generate_match_set(classifiers, perception)

        if previous_action_set is not None:
            acs2.apply_alp(classifiers,
                           action,
                           time,
                           action_set,
                           perception,
                           previous_perception)
            acs2.apply_rl(match_set, action_set, reward)
            acs2.apply_ga(classifiers, action_set, time)

        # Remove previous action set
        previous_action_set = None

        action = acs2.choose_action(match_set)
        action_set = acs2.generate_action_set(match_set, action)

        # Execute action and obtain reward
        reward = env.execute_action(action)

        # Next time slot
        time += 1
        previous_perception = perception
        previous_action_set = action_set

        perception = env.get_animat_perception()

        if time % 100 == 0:
            logger.info('=== 100 ===')
            # Some debug / measurements here

    logger.info('Finished')
