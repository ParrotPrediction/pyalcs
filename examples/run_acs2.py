import sys
import logging
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
    format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=logging.DEBUG)


if __name__ == '__main__':

    # Load environment
    env = Maze('mazes/MazeF1.maze')

    # Initialize agent
    agent = ACS2()

    # Specify which metrics will be collected
    agent.add_metrics_handlers([
        ActualStep('time'),
        SuccessfulTrial('found_reward'),
        ClassifierPopulationSize('total_classifiers'),
        AveragedFitnessScore('average_fitness'),
        AveragedConditionSpecificity('average_specificity'),
        AchievedKnowledge('achieved_knowledge')
    ])

    # Evaluate simulation
    classifiers, metrics = agent.evaluate(env, 500, 5)

    reliable = [c for c in classifiers if c.is_reliable()]

    print("Reliable classifiers: {}".format(len(reliable)))
