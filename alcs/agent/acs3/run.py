import logging

from alcs.environment.maze.Maze import Maze

from alcs.agent.acs3 import Constants as c
from alcs.agent.acs3 import ClassifiersList

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=logging.INFO)


env = Maze('mazes/MazeF4.maze')

def start_experiments():
    """
    Runs all the experiments
    """
    for experiment in range(0, c.NUMBER_OF_EXPERIMENTS):
        start_experiment()


def start_experiment():
    trial = 0
    all_steps = 0
    knowledge = 0

    population = ClassifiersList()

    while all_steps < c.MAX_STEPS:
        logger.info("Trial/steps: [{}/{}]".format(trial, all_steps))
        all_steps += start_one_trial_explore(population, env, all_steps)
        trial += 1


def start_one_trial_explore(population: ClassifiersList, env: Maze, time: int):
    steps = 0
    env.insert_animat()

    action = None
    reward = None
    previous_situation = None
    action_set = ClassifiersList()

    while not env.animat_found_reward and time + steps < c.MAX_STEPS and steps < c.MAX_TRIAL_STEPS:
        logger.debug("\nExecuting step: {}".format(steps))
        situation = env.get_animat_perception()
        match_set = ClassifiersList.form_match_set(population, situation)

        if steps > 0:
            # Apply learning in the last action set
            action_set.apply_alp(previous_situation, action, situation, time+steps, population, match_set)
            action_set.apply_reinforcement_learning(reward, match_set.get_maximum_fitness())
            action_set.apply_ga(time+steps, population, match_set, situation)

        action = match_set.choose_action(epsilon=c.EPSILON)
        action_set = ClassifiersList.form_action_set(match_set, action)

        reward = env.execute_action(action)

        previous_situation = situation
        situation = env.get_animat_perception()

        if env.animat_found_reward:
            action_set.apply_alp(previous_situation, action, situation, time+steps, population, None)
            action_set.apply_reinforcement_learning(reward, 0)
            action_set.apply_ga(time+steps, population, None, situation)

        steps += 1

    return steps

### FLOW STARTS HERE

start_experiments()

