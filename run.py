from environment import Environment, Maze, MazeAction

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    env = Maze('mazes/m1.maze')
    reward = env.execute_action(MazeAction.LEFT)
    logger.debug('Reward: %d', reward)
    logger.info("Finished")
