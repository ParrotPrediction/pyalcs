import logging

from .go_naive import Position, IllegalMove
from .go_naive import WHITE, swap_colors


class GoBoard:
    """
    Class mimic OpenAI Gym Environment interface but enables ACS2 to play
    with the same agent.
    """
    def __init__(self):
        self.pos = None
        self.color = None
        self.moves = 0

    def reset(self):
        logging.debug("Resetting Go board state")
        self.pos = Position.initial_state()
        self.color = None
        self.moves = 0
        return self.pos.get_board()

    def step(self, action):
        reward = 0
        done = False
        self._distinguish_player_color()
        logging.debug("[{}] moves to [{}]".format(self.color, action))

        try:
            self.pos = self.pos.play_move(action, self.color)
            reward = 0
            if not self._any_moves_left():
                reward = 1000
                done = True

        except IllegalMove:
            # logging.error("Illegal move exception, finishing trial")
            reward = -1
            done = True

        self.moves += 1

        return self.pos.get_board(), reward, done, self._load_debug_data()

    def _load_debug_data(self):
        return {
            'color': self.color,
            'move': self.moves,
            'score': self.pos.score()
        }

    def _distinguish_player_color(self):
        if self.color is not None:
            self.color = swap_colors(self.color)
        else:
            self.color = WHITE

    def _any_moves_left(self):
        return self.pos.can_move()
