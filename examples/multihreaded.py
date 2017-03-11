from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import sys
import logging
import time

from os.path import abspath, join, dirname

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from alcs.agent.acs2 import ACS2
from alcs.environment.maze import Maze
from alcs.helpers.metrics import \
    ActualStep,\
    ClassifierPopulationSize,\
    AveragedFitnessScore,\
    SuccessfulTrial,\
    AveragedConditionSpecificity,\
    AchievedKnowledge

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(threadName)s]: %(message)s',
    level=logging.WARN)

PROCESSES = 2
EXPERIMENTS = 20
STEPS = 2000  # 10 000
MAX_STEPS_IN_TRIAL = 50
MAZE_LOCATION = 'mazes/MazeF2.maze'


def perform_experiment(experiment):
    print('Performing experiment [{}]'.format(experiment))

    agent = ACS2()
    agent.add_metrics_handlers([
        ActualStep('time'),
        SuccessfulTrial('found_reward'),
        ClassifierPopulationSize('total_classifiers'),
        AveragedFitnessScore('average_fitness'),
        AveragedConditionSpecificity('average_specificity'),
        AchievedKnowledge('achieved_knowledge')
    ])

    # Re-initialize the environment
    env = Maze(MAZE_LOCATION)

    # Evaluate algorithm
    classifiers, metrics = agent.evaluate(env, STEPS, MAX_STEPS_IN_TRIAL)

    # Add information about the experiment into metrics
    metrics['experiment_id'] = [experiment] * len(metrics['time'])

    return classifiers, metrics


if __name__ == '__main__':
    all_classifiers = []
    all_metrics = pd.DataFrame()

    start = time.time()

    with ProcessPoolExecutor(PROCESSES) as executor:
        futures = []

        for i in range(EXPERIMENTS):
            future = executor.submit(perform_experiment, i)
            futures.append(future)

        for idx, el in enumerate(as_completed(futures)):
            classifiers, metrics = el.result()

            all_classifiers.append(classifiers)
            all_metrics = all_metrics.append(pd.DataFrame(metrics))

    end = time.time()

    print("\nTook {:.2f}s using {} processes".format(end-start, PROCESSES))

