from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import sys
import logging
import time

from os.path import abspath, join, dirname

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from alcs.agent.acs2 import ACS2, Classifier
from alcs.environment.maze import Maze
from alcs.helpers.metrics import \
    ActualStep,\
    ClassifierPopulationSize,\
    ReliableClassifierPopulationSize,\
    AveragedFitnessScore,\
    SuccessfulTrial,\
    AveragedConditionSpecificity,\
    AchievedKnowledge

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(threadName)s]: %(message)s',
    level=logging.WARN)

PROCESSES = 1
EXPERIMENTS = 1
STEPS = 10000  # 10
MAZE_LOCATION = 'mazes/MazeF2.maze'
EXPLOITATION_MODE = False


def perform_experiment(experiment):
    print('Performing experiment [{}]'.format(experiment))

    agent = ACS2(exploitation_mode=EXPLOITATION_MODE)

    agent.add_metrics_handlers([
        ActualStep('time'),
        SuccessfulTrial('found_reward'),
        ClassifierPopulationSize('total_classifiers'),
        ReliableClassifierPopulationSize('reliable_classifiers'),
        AveragedFitnessScore('average_fitness'),
        AveragedConditionSpecificity('average_specificity'),
        AchievedKnowledge('achieved_knowledge')
    ])

    # Re-initialize the environment
    env = Maze(MAZE_LOCATION)

    # Evaluate algorithm
    classifiers, metrics = agent.evaluate(env, STEPS)

    print("Total number of macro-classifiers: {}"
          .format(len(classifiers)))
    print("Total number of classifiers: {}"
          .format(sum(cl.num for cl in classifiers)))
    print("Marked classifiers: {}"
          .format(sum(1 for cl in classifiers
                      if Classifier.is_marked(cl.mark))))
    print("Achieved knowledge: {:.2f}%"
          .format(metrics["achieved_knowledge"][-1] * 100))

    # Add information about the experiment into metrics
    metrics['experiment_id'] = [experiment] * len(metrics['time'])

    # Print classifiers
    reliable = [cls for cls in classifiers if cls.is_reliable()]
    reliable.sort(key=lambda cls: cls.action)

    for cls in reliable:
        print(cls)

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

    print("\nTook {:.2f}s using {} processes".format(end - start, PROCESSES))
