from enum import Enum, unique


@unique
class MazeAction(Enum):
    LEFT = 0
    TOP = 1
    RIGHT = 2
    DOWN = 3