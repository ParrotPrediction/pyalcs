import logging


def moves_9x9():
    """
    Return a list of all available moves (such as A8) on 9x9 board.
    """
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J']
    cols = list(range(1, 9 + 1))
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


def calculate_bw_ratio(env):
    """
    Calculates the ration between black/white stones.
    """
    total_blacks = env.state.board.black_stones.size
    total_whites = env.state.board.white_stones.size

    ratio = 0.0

    try:
        ratio = total_blacks / total_whites
    except ZeroDivisionError:
        # Whites did not moved yet
        pass

    return ratio
