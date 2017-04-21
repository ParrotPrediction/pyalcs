from alcs.environment.maze.Maze import Maze

from alcs.agent.acs3 import Constants as c
from alcs.agent.acs3.ClassifiersList import ClassifiersList


env = Maze('mazes/MazeF1.maze')

# Make sure that perception length is set
# Make sure that environment actions are set


def start_experiments():
    """
    Runs all the experiments
    """
    for experiment in range(0, c.NUMBER_OF_EXPERIMENTS):
        start_experiment()


def start_experiment():
    print("Running experiment")
    all_steps = 0
    trial = 0
    knowledge = 0

    population = ClassifiersList()

    while all_steps < c.MAX_STEPS:
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
        situation = env.get_animat_perception()
        match_set = population.form_match_set(situation)

        if steps > 0:
            # Apply learning in the last action set
            action_set.apply_alp(previous_situation, action, situation, time+steps, population, match_set)
            action_set.apply_reinforcement_learning(reward, match_set.get_maximum_fitness())
            action_set.apply_ga(time+steps, population, match_set, situation)

        # ...
        steps += 1

    return steps

### FLOW STARTS HERE

start_experiments()

