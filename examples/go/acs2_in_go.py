import logging

from alcs import ACS2, ACS2Configuration

import gym
from gym.envs.board_game import go

logging.basicConfig(level=logging.DEBUG)


def moves_9x9():
    """
    Return a list of all available moves (such as A8) on 9x9 board.
    """
    rows = [chr(i) for i in range(ord('A'), ord('H') + 1)]
    cols = list(range(1, 8 + 1))
    actions = []

    for row in rows:
        for col in cols:
            actions.append("{}{}".format(row, col))

    return actions


def process_state(state):
    """
    Returns a flatten array of board state of given state.
    Black stones are represented as 'B', whites as 'W'
    """
    black_stones = state[0]
    white_stones = state[1] * 2

    board = (black_stones + white_stones).astype('str')
    board[board == '1'] = 'B'
    board[board == '2'] = 'W'

    return list(board.flatten())

if __name__ == '__main__':
    env = gym.make('Go9x9-v0')
    state = env.reset()

    moves = {move: go.str_to_action(env._state.board, move) for move in
             moves_9x9()}

    CLASSIFIER_LENGTH = env._state.board.size ** 2
    NUMBER_OF_POSSIBLE_ACTIONS = len(moves)

    cfg = ACS2Configuration(
        classifier_length=CLASSIFIER_LENGTH,
        number_of_possible_actions=NUMBER_OF_POSSIBLE_ACTIONS,
        perception_mapper_fcn=process_state
    )

    logging.info(cfg)

    # Create the agent
    agent = ACS2(cfg)

    population, metrics = agent.explore(env, 200)

    logging.info("Done")
