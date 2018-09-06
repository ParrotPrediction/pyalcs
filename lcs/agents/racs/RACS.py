import logging
from typing import Optional

from ...agents import Agent
from ...agents.Agent import Metric
from ...agents.racs import Configuration, ClassifierList
from ...utils import parse_state


logger = logging.getLogger(__name__)


class RACS(Agent):
    """ACS2 agent operating on real-valued (floating) number"""

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifierList=None) -> None:
        self.cfg = cfg
        self.population = population or ClassifierList()

    def explore(self, env, trials):
        pass

    def exploit(self, env, trials):
        pass

    def _run_trial_explore(self, env, time):
        logger.debug("** Running trial explore ** ")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = parse_state(raw_state)
        action = None
        reward = None
        prev_state = None
        action_set = ClassifierList()
        done = False

        # TODO: rest of the code

    def _collect_agent_metrics(self, trial, steps, total_steps) -> Metric:
        return {
            "population": 0
        }

    def _collect_environment_metrics(self, env) -> Optional[Metric]:
        return None

    def _collect_performance_metrics(self, env) -> Optional[Metric]:
        return None
