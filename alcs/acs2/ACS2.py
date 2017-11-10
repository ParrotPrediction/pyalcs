import logging
from alcs.acs2 import ClassifiersList, ACS2Configuration


class ACS2:
    def __init__(self, cfg: ACS2Configuration, population=None):
        self.cfg = cfg
        self.population = population or ClassifiersList(cfg=self.cfg)

    def explore(self, env, max_trials):
        current_trial = 0
        steps = 0

        metrics = []
        while current_trial < max_trials:
            steps_in_trial = self._run_trial_explore(env, steps)
            steps += steps_in_trial

            # Collect metrics of trial
            agent_stats = self._collect_agent_metrics(
                current_trial, steps_in_trial, steps)

            if self.cfg.environment_metrics_fcn:
                env_stats = self.cfg.environment_metrics_fcn(env)

            # TODO: add env stats
            metrics.append(agent_stats)

            current_trial += 1

        return self.population, metrics

    def exploit(self, env, max_trials):
        current_trial = 0
        steps = 0

        metrics = []
        while current_trial < max_trials:
            steps_in_trial = self._run_trial_exploit(env)
            steps += steps_in_trial

            # Collect metrics of trial
            metrics.append(self._collect_agent_metrics(
                current_trial, steps_in_trial, steps))

            current_trial += 1

        return self.population, metrics

    def _run_trial_explore(self, env, time):
        logging.debug("Running trial explore")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self._parse_state(raw_state)
        logging.debug("Initial state: [%s]", ''.join(state))

        action = None
        reward = None
        prev_state = None
        action_set = ClassifiersList(cfg=self.cfg)
        done = False

        while not done:
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
            logging.debug("Executing action: [%d]", action)
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

    def _run_trial_exploit(self, env):
        logging.debug("Running trial explore")
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
