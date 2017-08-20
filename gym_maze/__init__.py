import logging

from gym.envs.registration import register

from gym_maze.Maze import Maze
from gym_maze.Maze import PATH_MAPPING, WALL_MAPPING, REWARD_MAPPING

logger = logging.getLogger(__name__)

register(
    id='Maze1-v0',
    entry_point='gym_maze.envs:Maze1',
    nondeterministic=True
)
