import logging
from random import choice

import gym

# noinspection PyUnresolvedReferences
import gym_maze

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    maze = gym.make('MazeF1-v0')

    possible_actions = list(range(8))
    transitions = maze.env.get_all_possible_transitions()

    for i_episode in range(1):
        observation = maze.reset()

        for t in range(100):
            logging.info("Time: [{}], observation: [{}]".format(t, observation))

            action = choice(possible_actions)

            logging.info("\t\tExecuted action: [{}]".format(action))
            observation, reward, done, info = maze.step(action)

            if done:
                logging.info("Episode finished after {} timesteps.".format(t + 1))
                logging.info("Last reward: {}".format(reward))
                break

    logging.info("Finished")
