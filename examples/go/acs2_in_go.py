import logging

import gym

from alcs import ACS2, ACS2Configuration
from examples.go.utils import moves_9x9, process_state, \
    calculate_bw_ratio, map_moves

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    filename='go.log',
    filemode='w'
)

if __name__ == '__main__':
    env = gym.make('Go9x9-v0')
    state = env.reset()

    # Create a mapping dictionary of moves
    # A key is number 0,1,...num_moves, and the key is corresponding
    # action in Pachi environment
    moves = {idx: map_moves(env, move) for idx, move in enumerate(moves_9x9())}

    # Define a function for obtaining environment metrics
    def environment_metrics(env):
        return {
            'bw_ratio': calculate_bw_ratio(env)
        }

    CLASSIFIER_LENGTH = env._state.board.size ** 2
    NUMBER_OF_POSSIBLE_ACTIONS = len(moves)

    cfg = ACS2Configuration(
        classifier_length=CLASSIFIER_LENGTH,
        number_of_possible_actions=NUMBER_OF_POSSIBLE_ACTIONS,
        perception_mapper_fcn=process_state,
        environment_metrics_fcn=environment_metrics,
        action_mapping_dict=moves,
        epsilon=0.4
    )

    logging.info(cfg)

    # Create the agent
    agent = ACS2(cfg)

    population, metrics = agent.explore(env, 500)

    logging.info("Done")
