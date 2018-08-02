from typing import Optional

from ...agents import Agent
from ...agents.Agent import Metric


class RACS(Agent):
    """ACS2 agent operating on real-valued (floating) number"""

    def explore(self, env, trials):
        pass

    def exploit(self, env, trials):
        pass

    def _collect_agent_metrics(self, trial, steps, total_steps) -> Metric:
        return {
            "population": 0
        }

    def _collect_environment_metrics(self, env) -> Optional[Metric]:
        return None

    def _collect_performance_metrics(self, env) -> Optional[Metric]:
        return None
