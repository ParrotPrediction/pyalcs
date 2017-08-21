import logging

from gym.envs.registration import register

from gym_maze.Maze import Maze
from gym_maze.Maze import PATH_MAPPING, WALL_MAPPING, REWARD_MAPPING

logger = logging.getLogger(__name__)

register(
    id='MazeF1-v0',
    entry_point='gym_maze.envs:MazeF1',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeF2-v0',
    entry_point='gym_maze.envs:MazeF2',
    max_episode_steps=50,
    nondeterministic=False
)
