import pickle

# Logger
import logging
logging.basicConfig(level=logging.INFO)

# Load PyALCS module
from alcs import ACS2, ACS2Configuration

# Load environments
import gym
import gym_multiplexer

from gym_multiplexer.utils import get_correct_answer


def evaluate_performance(env, population, ctrl_bits):
    p1 = env.render()  # state after executing action
    p0 = p1[:-1] + '0'  # initial state
    true_action = get_correct_answer(p0, ctrl_bits)

    # get all classifiers matching initial state
    matching_cls = {c for c in population if c.condition.does_match(p0)}
    best_cl = max(matching_cls, key=lambda cl: cl.q)

    return {'was_correct': best_cl.predicts_successfully(p0, true_action, p1)}


def get_actors():
    mp = gym.make('boolean-multiplexer-37bit-v0')
    cfg = ACS2Configuration(
        mp.env.observation_space.n,
        2,
        performance_fcn=evaluate_performance,
        performance_fcn_params={'ctrl_bits': 5},
        do_ga=True)

    return ACS2(cfg), mp


def perform_experiment(agent, env, trials=50):
    population, metrics = agent.explore_exploit(env, trials)
    print("Population size: {}".format(metrics[-1]['agent']['population']))
    print("Reliable size: {}".format(metrics[-1]['agent']['reliable']))
    print(metrics[-1])

    reliable_classifiers = [c for c in population if c.is_reliable()]
    reliable_classifiers = sorted(reliable_classifiers, key=lambda cl: -cl.q)

    # Print top 20 reliable classifiers
    for cl in reliable_classifiers[:20]:
        print(f"{cl}, q: {cl.q:.2f}, exp: {cl.exp:.2f}")

    return population, metrics


# Run experiment
TRIALS = 250_000
population, metrics = perform_experiment(*get_actors(), trials=TRIALS)

# Dump data to file
logging.info("Dumping data to files")
pickle.dump(population, open("population.p", "wb"))
pickle.dump(metrics, open("metrics.p", "wb"))

logging.info("Experiment completed")
