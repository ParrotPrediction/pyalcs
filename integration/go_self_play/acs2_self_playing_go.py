from integration.go_self_play.environment import GoBoard
from lcs import ACS2, ACS2Configuration

import logging

# Configure logger
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':

    # Initialize board of size 9x9
    board = GoBoard()

    # Configure ACS2 agents
    cfg = ACS2Configuration(81, 81, do_ga=True)
    white_agent = ACS2(cfg)
    black_agent = ACS2(cfg)

    # Play some games
    MAX_GAMES = 2
    game = 0

    while game < MAX_GAMES:
        move = 0
        board.reset()

        if move % 2 == 0:
            white_agent.explore(board, None)

        move += 1
        game += 1

    # Random step
    state, reward, done, debug = board.step(12)
    state, reward, done, debug = board.step(13)

    print(state)
    print(reward)
    print(done)
    print(debug)
