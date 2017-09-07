from random import choice

import logging

import gym
import gym_maze

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

env = gym.make('BMaze4-v0')

possible_actions = list(range(8))

for i_episode in range(1):
    observation = env.reset()

    for t in range(100):
        logger.info("Time: [{}], observation: [{}]".format(t, observation))

        action = choice(possible_actions)

        logger.info("\t\tExecuted action: [{}]".format(action))
        observation, reward, done, info = env.step(action)

        if done:
            logger.info("Episode finished after {} timesteps.".format(t + 1))
            logger.info("Last reward: {}".format(reward))
            break

logger.info("Finished")
