from typing import Dict, Any, Optional

Metric = Dict[str, Any]


class Agent:
    def explore(self, env, trials):
        raise NotImplementedError()

    def exploit(self, env, trials):
        raise NotImplementedError()

    def _collect_agent_metrics(self, trial, steps, total_steps) -> Metric:
        raise NotImplementedError()

    def _collect_environment_metrics(self, env) -> Optional[Metric]:
        raise NotImplementedError()

    def _collect_performance_metrics(self, env, reward) -> Optional[Metric]:
        raise NotImplementedError()

    def _collect_metrics(self,
                         env,
                         current_trial: int,
                         steps_in_trial: int,
                         steps: int,
                         reward: int) -> Dict[str, Optional[Metric]]:
        """

        Parameters
        ----------
        env
            current environment
        current_trial: int
            trial the agent is currently executing
        steps_in_trial: int
            steps in given trial
        steps:
            total steps so far (all trials)
        reward: int
            final reward obtained

        Returns
        -------
        Dict[str, Metric]
        """
        return {}
        # agent_stats = self._collect_agent_metrics(
        #     current_trial, steps_in_trial, steps)
        # env_stats = self._collect_environment_metrics(env)
        # performance_stats = self._collect_performance_metrics(env, reward)
        #
        # return {
        #     'agent': agent_stats,
        #     'environment': env_stats,
        #     'performance': performance_stats
        # }
