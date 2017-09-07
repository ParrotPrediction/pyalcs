import sys
import logging

from alcs.agent import ACS2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import gym
import gym_maze

from os.path import abspath, join, dirname
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

if __name__ == '__main__':

    # Load desired environment
    env = gym.make('MazeF3-v0')

    # Create the agent
    agent = ACS2()

    # Explore the environment
    logger.info("EXPLORE PHASE")
    population, metrics = agent.explore(env)
    logger.info(metrics)

    # Exploit the environment
    logger.info("EXPLOIT PHASE")
    state = env.reset()

    logger.info("Before: {}".format(state))
    env.render()

    action = agent.exploit(state)
    logger.info("Action chosen: {}".format(action))

    state, reward, done, _ = env.step(action)

    logger.info("After: {}".format(state))
    env.render()
