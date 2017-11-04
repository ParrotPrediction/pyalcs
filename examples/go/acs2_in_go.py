# https://github.com/openai/gym/blob/master/examples/scripts/play_go

from alcs import ACS2

import gym
from gym.envs.board_game import go

if __name__ == '__main__':

    # Load desired environment
    env = gym.make('Go9x9-v0')
    env.reset()

    s = env._state
    env.render()
    a = go.str_to_action(s.board, "A8")
    print("oo")
