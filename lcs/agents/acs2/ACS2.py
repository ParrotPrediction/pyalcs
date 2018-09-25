import logging
from typing import Optional

from lcs.strategies.action_planning.action_planning import \
    search_goal_sequence, exists_classifier
from . import ClassifiersList, Configuration
from ...agents import Agent
from ...agents.Agent import Metric
from ...strategies.action_selection import choose_action
from ...utils import parse_state, parse_action


class ACS2(Agent):
    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None) -> None:
        self.cfg = cfg

        if population:
            self.population = population
        else:
            self.population = ClassifiersList(cfg=self.cfg)

    def explore(self, env, trials):
        """
        Explores the environment in given set of trials.
        :param env: environment
        :param trials: number of trials
        :return: population of classifiers and metrics
        """
        return self._evaluate(env, trials, self._run_trial_explore)

    def exploit(self, env, trials):
        """
        Exploits the environments in given set of trials (always executing
        best possible action - no exploration).
        :param env: environment
        :param trials: number of trials
        :return: population of classifiers and metrics
        """
        return self._evaluate(env, trials, self._run_trial_exploit)

    def explore_exploit(self, env, trials):
        """
        Alternates between exploration and exploitation phases.
        :param env: environment
        :param trials: number of trials
        :return: population of classifiers and metrics
        """

        def switch_phases(env, steps, current_trial):
            if current_trial % 2 == 0:
                return self._run_trial_explore(env, steps)
            else:
                return self._run_trial_exploit(env, None)

        return self._evaluate(env, trials, switch_phases)

    def _evaluate(self, env, max_trials, func):
        """
        Runs the classifier in desired strategy (see `func`) and collects
        metrics.

        Parameters
        ----------
        env:
            OpenAI Gym environment
        max_trials: int
            maximum number of trials
        func: Callable
            Function accepting three parameters: env, steps already made,
             current trial

        Returns
        -------
        tuple
            population of classifiers and metrics
        """
        current_trial = 0
        steps = 0

        metrics = []
        while current_trial < max_trials:
            steps_in_trial = func(env, steps, current_trial)
            steps += steps_in_trial

            trial_metrics = self._collect_metrics(
                env, current_trial, steps_in_trial, steps)
            logging.info(trial_metrics)
            metrics.append(trial_metrics)

            current_trial += 1

        return self.population, metrics

    def _run_trial_explore(self, env, time, current_trial=None):
        logging.debug("** Running trial explore ** ")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = parse_state(raw_state, self.cfg.perception_mapper_fcn)
        action = None
        reward = None
        prev_state = None
        action_set = ClassifiersList(cfg=self.cfg)
        done = False

        while not done:
            if self.cfg.do_action_planning and \
                    self._time_for_action_planning(steps + time):
                # Action Planning for increased model learning
                steps_ap, state, prev_state, action_set, reward = \
                    self._run_action_planning(env, steps + time, state,
                                              prev_state, action_set, action,
                                              reward)
                steps += steps_ap

            match_set = self.population.form_match_set(state,
                                                       self.cfg)

            if steps > 0:
                # Apply learning in the last action set
                action_set.apply_alp(
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.population,
                    match_set)
                action_set.apply_reinforcement_learning(
                    reward,
                    match_set.get_maximum_fitness())
                if self.cfg.do_ga:
                    action_set.apply_ga(
                        time + steps,
                        self.population,
                        match_set,
                        state)

            action = choose_action(
                match_set,
                self.cfg.number_of_possible_actions,
                self.cfg.epsilon)
            internal_action = parse_action(action, self.cfg.action_mapping_fcn)
            logging.debug("\tExecuting action: [%d]", action)
            action_set = match_set.form_action_set(action, self.cfg)

            prev_state = state
            raw_state, reward, done, _ = env.step(internal_action)
            state = parse_state(raw_state, self.cfg.perception_mapper_fcn)

            if done:
                action_set.apply_alp(
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.population,
                    None)
                action_set.apply_reinforcement_learning(
                    reward,
                    0)
            if self.cfg.do_ga:
                action_set.apply_ga(
                    time + steps,
                    self.population,
                    None,
                    state)

            steps += 1

        return steps

    def _run_trial_exploit(self, env, time=None, current_trial=None):
        logging.debug("** Running trial exploit **")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = parse_state(raw_state, self.cfg.perception_mapper_fcn)

        reward = None
        action_set = ClassifiersList(cfg=self.cfg)
        done = False

        while not done:
            match_set = self.population.form_match_set(state, self.cfg)

            if steps > 0:
                action_set.apply_reinforcement_learning(
                    reward,
                    match_set.get_maximum_fitness())

            # Here while exploiting always choose best action
            action = choose_action(
                match_set,
                self.cfg.number_of_possible_actions,
                epsilon=0.0)
            internal_action = parse_action(action, self.cfg.action_mapping_fcn)
            action_set = match_set.form_action_set(action, self.cfg)

            raw_state, reward, done, _ = env.step(internal_action)
            state = parse_state(raw_state, self.cfg.perception_mapper_fcn)

            if done:
                action_set.apply_reinforcement_learning(reward, 0)

            steps += 1

        return steps

    def _run_action_planning(self, env,
                             time: int,
                             situation: str,
                             previous_situation: str,
                             action_set: ClassifiersList,
                             action: int,
                             reward: int):
        """
        Executes action planning for model learning speed up.
        Method requests goals from 'goal generator' provided by
        the environment. If goal is provided, ACS2 searches for
        a goal sequence in the current model (only the reliable classifiers).
        This is done as long as goals are provided and ACS2 finds a sequence
        and successfully reaches the goal.
        :param env:
        :param time:
        :param situation:
        :param previous_situation:
        :param action_set:
        :param action:
        :param reward:
        :return:
        """
        logging.debug("** Running action planning **")

        # The environment has to have a function "get_goal_state"
        if not hasattr(env.env, "get_goal_state"):
            logging.debug("Action planning stopped - "
                          "no function get_goal_state in env")
            return 0, situation, previous_situation, action_set, reward

        steps = 0
        done = False

        while not done:
            goal_situation = env.env.get_goal_state()

            if goal_situation is None:
                break

            act_sequence = search_goal_sequence(self.population, situation,
                                                goal_situation)

            # Execute the found sequence and learn during executing
            i = 0
            for act in act_sequence:
                if act == -1:
                    break

                match_set = self.population.form_match_set(situation=situation,
                                                           cfg=self.cfg)
                if action_set is not None and previous_situation is not None:
                    action_set.apply_alp(previous_situation, action, situation,
                                         time + steps, self.population,
                                         match_set)
                    action_set.\
                        apply_reinforcement_learning(reward,
                                                     match_set.
                                                     get_maximum_fitness())
                    if self.cfg.do_ga:
                        action_set.apply_ga(time + steps, self.population,
                                            match_set, situation)

                action = act
                action_set = ClassifiersList.form_action_set(match_set, action,
                                                             self.cfg)

                raw_state, reward, done, _ = env.step(parse_action(action))
                previous_situation = situation
                situation = parse_state(raw_state)

                if not exists_classifier(action_set, previous_situation,
                                         action, situation, self.cfg.theta_r):
                    # no reliable classifier was able to anticipate
                    # such a change
                    break

                steps += 1
                i += 1

            if i == 0:
                break

        return steps, situation, previous_situation, action_set, reward

    def _time_for_action_planning(self, time):
        return time % self.cfg.action_planning_frequency == 0

    def _collect_agent_metrics(self, trial, steps, total_steps) -> Metric:
        return {
            'population': len(self.population),
            'numerosity': sum(cl.num for cl in self.population),
            'reliable': len([cl for cl in
                             self.population if cl.is_reliable()]),
            'fitness': (sum(cl.fitness for cl in self.population) /
                        len(self.population)),
            'trial': trial,
            'steps': steps,
            'total_steps': total_steps
        }

    def _collect_environment_metrics(self, env) -> Optional[Metric]:
        if self.cfg.environment_metrics_fcn:
            return self.cfg.environment_metrics_fcn(env)

        return None

    def _collect_performance_metrics(self, env) -> Optional[Metric]:
        if self.cfg.performance_fcn:
            return self.cfg.performance_fcn(
                env, self.population, **self.cfg.performance_fcn_params)

        return None
