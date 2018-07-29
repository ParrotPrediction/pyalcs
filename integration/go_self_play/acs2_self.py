# import logging
from lcs import ACS2Configuration
from lcs.agents.acs2 import ClassifiersList
from integration.go_self_play.environment import GoBoard

# Configure logger
# logging.basicConfig(level=logging.INFO)

GAMES = 5000  # How many games to play
ALL_MOVES = 0

# Commons
board = GoBoard()  # Initialize board of size 9x9
cfg = ACS2Configuration(81, 81, epsilon=0.6, do_ga=True)
population = ClassifiersList(cfg=cfg)


def determine_player(moves):
    """Returns current player mark based on the move number"""
    return ['W', 'B'][moves % 2]


def switch_perception(perception):
    return perception \
        .replace('O', 't') \
        .replace('X', 'O') \
        .replace('t', 'X')


def print_metrics(game, moves, population):
    print("Game [{}] finished".format(game))
    print("Total moves: [{}]".format(moves))
    print("Population: [{}/{}/{}r]".format(
        len(population),
        sum(cl.num for cl in population),
        len([cl for cl in population if cl.is_reliable()]))
    )


if __name__ == '__main__':

    # Play some games
    for g in range(GAMES):
        action_set = ClassifiersList(cfg=cfg)
        prev_state, action, reward, done = None, None, None, False
        state = board.reset()
        moves = 0

        while not done:
            player = determine_player(moves)  # Determine player

            match_set = ClassifiersList.form_match_set(population, state, cfg)

            if moves > 0:
                action_set.apply_alp(
                    prev_state,
                    action,
                    state,
                    ALL_MOVES + moves,
                    population,
                    match_set)
                action_set.apply_reinforcement_learning(
                    reward, match_set.get_maximum_fitness())
                if cfg.do_ga:
                    action_set.apply_ga(
                        ALL_MOVES + moves, population, match_set, state)

            # Determine best action
            action = match_set.choose_action(cfg.epsilon)

            action_set = ClassifiersList.form_action_set(
                match_set, action, cfg)

            prev_state = state
            state, reward, done, debug = board.step(action)

            if done:
                action_set.apply_alp(
                    prev_state,
                    action,
                    state,
                    ALL_MOVES + moves,
                    population,
                    None)
                action_set.apply_reinforcement_learning(reward, 0)

                if cfg.do_ga:
                    action_set.apply_ga(
                        ALL_MOVES + moves, population, None, state)

                if g % 10 == 0:
                    print_metrics(g, ALL_MOVES, population)

            # For next player let's change perception
            state = switch_perception(state)
            prev_state = switch_perception(prev_state)

            moves += 1
            ALL_MOVES += 1

    print("OK")
