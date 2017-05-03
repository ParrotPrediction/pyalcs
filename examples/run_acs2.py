import sys
import logging
from os.path import abspath, join, dirname

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from alcs.agent.acs2 import ACS2
from alcs.environment.maze import Maze
from alcs.helpers.metrics import \
    StepsInTrial, \
    Experiment, \
    ClassifierPopulationSize, \
    ReliableClassifierPopulationSize, \
    AchievedKnowledge

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=logging.INFO)


if __name__ == '__main__':

    # Load environment
    env = Maze('mazes/BMaze4.maze')

    # Initialize agent
    agent = ACS2()

    # Specify which metrics will be collected
    agent.add_metrics_handlers([
        StepsInTrial('steps'),
        Experiment('experiment_id'),
        ClassifierPopulationSize('total_classifiers'),
        ReliableClassifierPopulationSize('reliable_classifiers'),
        AchievedKnowledge('knowledge')
    ])

    # Evaluate simulation
    metrics = agent.evaluate(env, 1, 1000)

    logger.info("Algorithm finished")
    logger.info("Last knowledge: {:.2f}%".format(metrics['knowledge'][-1]))
