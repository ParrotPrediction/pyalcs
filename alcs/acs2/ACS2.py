import logging
from alcs.acs2 import ClassifiersList, ACS2Configuration


class ACS2:
    def __init__(self, cfg: ACS2Configuration, population=None):
        self.cfg = cfg
        self.population = population or ClassifiersList(cfg=self.cfg)

    def explore(self, env, max_trials):
        """
        Explores the environment in given set of trials.
        :param env: environment
        :param max_trials: number of trials
        :return: population of classifiers and metrics
        """
        return self._evaluate(env, max_trials, self._run_trial_explore)

    def exploit(self, env, max_trials):
        """
        Exploits the environments in given set of trials (always executing
        best possible action - no exploration).
        :param env: environment
        :param max_trials: number of trials
        :return: population of classifiers and metrics
        """
        return self._evaluate(env, max_trials, self._run_trial_exploit)

    def explore_exploit(self, env, max_trials):
        """
        Alternates between exploration and exploitation phases.
        :param env: environment
        :param max_trials: number of trials
        :return: population of classifiers and metrics
        """

        def switch_phases(env, steps, current_trial):
            if current_trial % 2 == 0:
                return self._run_trial_explore(env, steps)
            else:
                return self._run_trial_exploit(env, None)

        return self._evaluate(env, max_trials, switch_phases)

    def _evaluate(self, env, max_trials, func):
        """
        Runs the classifier in desired strategy (see `func`) and collects
        metrics.
        :param env: environment
        :param max_trials: number of trials
        :param func: three arguments: env, steps already made, current trial
        :return: population of classifiers and metrics
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
        state = self._parse_state(raw_state)
        action = None
        reward = None
        prev_state = None
        action_set = ClassifiersList(cfg=self.cfg)
        done = False

        while not done:
            if self.cfg.do_action_planning and (steps + time) % self.cfg.action_planning_frequency == 0:
                # TODO: check if HandEye?
                self._run_action_planning(env, steps + time, state, prev_state, action_set, action, reward)

            match_set = ClassifiersList.form_match_set(self.population,
                                                       state,
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

            action = match_set.choose_action(self.cfg.epsilon)
            logging.debug("\tExecuting action: [%d]", action)
            action_set = ClassifiersList.form_action_set(match_set,
                                                         action,
                                                         self.cfg)

            prev_state = state
            raw_state, reward, done, _ = env.step(self._parse_action(action))
            state = self._parse_state(raw_state)

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
        state = self._parse_state(raw_state)

        reward = None
        action_set = ClassifiersList(cfg=self.cfg)
        done = False

        while not done:
            match_set = ClassifiersList.form_match_set(self.population,
                                                       state,
                                                       self.cfg)

            if steps > 0:
                action_set.apply_reinforcement_learning(
                    reward,
                    match_set.get_maximum_fitness())

            # Here while exploiting always choose best action
            action = match_set.choose_action(epsilon=0.0)
            action_set = ClassifiersList.form_action_set(match_set,
                                                         action,
                                                         self.cfg)

            raw_state, reward, done, _ = env.step(self._parse_action(action))
            state = self._parse_state(raw_state)

            if done:
                action_set.apply_reinforcement_learning(reward, 0)

            steps += 1

        return steps

    def _run_action_planning(self, env, time, situation,
                             previous_situation, action_set, action, reward):

        if not hasattr(env, "get_goal_state"):
            return 0

        steps = 0
        match_set = 0
        done = False
        goal_situation = 0

        while not done:
            goal_situation = env.get_goal_state()

            if goal_situation is None:
                return 0

            act_sequence = self.population.search_goal_sequence(situation, goal_situation) # TODO: search_goal_sequence (ClassifierList)
            i = 0
            while act_sequence[i] != 0:
                match_set = ClassifiersList(self.population, situation)  # TODO new constructor???
                if action_set is not None:
                    action_set.apply_alp(previous_situation, action, situation, time + steps, self.population, match_set)
                    action_set.apply_reinforcement_learning(reward, match_set.get_maximum_fitness())
                    if self.cfg.do_ga:
                        action_set.apply_ga(time + steps, self.population, match_set, situation)

                action = act_sequence[i]
                action_set = ClassifiersList(match_set, action)

                previous_situation = situation
                raw_state, reward, done, _ = env.step(self._parse_action(action))
                situation = self._parse_state(raw_state)

                if action_set.exists_classifier(previous_situation, action, situation, self.cfg.theta_r):
                    break

                i += 1
                steps += 1

            if not(i == 0 or action != 0): # TODO: necessary?
                break

        return steps

    def _parse_state(self, raw_state):
        """
        Sometimes the environment state returned by the OpenAI
        environment does not suit to the classifier representation
        of data used by ACS2. If a mapping function is defined in
        configuration - use it.

        :param raw_state: state obtained from OpenAI gym
        :return: state suitable for ACS2 (list)
        """
        if self.cfg.perception_mapper_fcn:
            return self.cfg.perception_mapper_fcn(raw_state)

        return raw_state

    def _parse_action(self, action_idx):
        """
        Sometimes the step function from OpenAI Gym takes different
        representation of actions than sequential range of integers.
        There is a possiblity to provide custom mapping function for
        suitable action values.

        :param action_idx: action id, used in ACS2
        :return: action id for the step function
        """
        if self.cfg.action_mapping_dict:
            return self.cfg.action_mapping_dict[action_idx]

        return action_idx

    def _collect_metrics(self, env, current_trial, steps_in_trial, steps):
        agent_stats = self._collect_agent_metrics(
            current_trial, steps_in_trial, steps)
        env_stats = self._collect_env_stats(env)
        performance_stats = self._calculate_performance(env)

        return {
            'agent': agent_stats,
            'environment': env_stats,
            'performance': performance_stats
        }

    def _collect_agent_metrics(self, trial, steps, total_steps):
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

    def _collect_env_stats(self, env):
        if self.cfg.environment_metrics_fcn:
            return self.cfg.environment_metrics_fcn(env)

        return None

    def _calculate_performance(self, env):
        if self.cfg.performance_fcn:
            return self.cfg.performance_fcn(
                env, self.population, **self.cfg.performance_fcn_params)

        return None
