import logging

import gym
import gym_maze

logger = logging.getLogger()
logger.setLevel(logging.INFO)

env = gym.make('Maze1-v0')

for i_episode in range(1):
    observation = env.reset()

    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break


logger.info("Finished")
