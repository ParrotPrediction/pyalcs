import logging
import pickle

import gym

from examples.acs2.go.utils import moves_9x9, process_state, \
    calculate_environment_metrics, map_moves
from lcs.agents.acs2 import ACS2, Configuration

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    filename='go.log',
    filemode='w'
)

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    env = gym.make('Go9x9-v0')  # TODO: removed from GYM after 0.9 :(
    state = env.reset()

    # Create a mapping dictionary of moves
    # A key is number 0,1,...num_moves, and the key is corresponding
    # action in Pachi environment
    moves = {idx: map_moves(env, move) for idx, move in enumerate(moves_9x9())}

    CLASSIFIER_LENGTH = env._state.board.size ** 2
    NUMBER_OF_POSSIBLE_ACTIONS = len(moves)

    cfg = Configuration(
        classifier_length=CLASSIFIER_LENGTH,
        number_of_possible_actions=NUMBER_OF_POSSIBLE_ACTIONS,
        perception_mapper_fcn=process_state,
        environment_metrics_fcn=calculate_environment_metrics,
        action_mapping_dict=moves,
        epsilon=0.4,
        do_ga=True
    )

    logger.info(cfg)

    # Create the agent
    agent = ACS2(cfg)
    population, metrics = agent.explore_exploit(env, 50)

    # Store metrics in file
    logger.info("Dumping data to files ...")
    pickle.dump(population, open("go_population.pkl", "wb"))
    pickle.dump(metrics, open("go_metrics.pkl", "wb"))

    logger.info("Done")
