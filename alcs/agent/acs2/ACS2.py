from alcs.agent.acs2 import ClassifiersList


class ACS2:
    def __init__(self, population=None):
        if population is not None:
            self.population = population
        else:
            self.population = ClassifiersList()

    def explore(self, env, max_steps=10000):
        metrics = self._run_experiment(env, max_steps)
        return self.population, metrics

    def exploit(self, state):
        """
        Exploits the environment returning the best possible action in given
        state
        :param state: observation (state) of the environment
        :return: integer describing the action
        """
        applicable_cls = [cl for cl in
                          self.population if cl.condition.does_match(state)]
        best_cl = max(applicable_cls, key=lambda cl: cl.fitness)

        return best_cl.action

    def _run_experiment(self, env, max_steps):
        trials = 0
        steps = 0

        while steps < max_steps:
            steps_in_trial = self._run_trial_explore(env, steps, max_steps)
            steps += steps_in_trial
            trials += 1

        return self._collect_metrics(trials, max_steps)

    def _run_trial_explore(self, env, time, max_steps):
        # Initial conditions
        steps = 0
        state = env.reset()

        action = None
        reward = None
        prev_state = None
        action_set = ClassifiersList()
        done = False

        while not done and time + steps <= max_steps:
            match_set = ClassifiersList.form_match_set(
                self.population, state)

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
                action_set.apply_ga(
                    time + steps,
                    self.population,
                    match_set,
                    state)

            action = match_set.choose_action(epsilon=1.0)
            action_set = ClassifiersList.form_action_set(match_set, action)

            prev_state = state
            state, reward, done, _ = env.step(action)

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
                action_set.apply_ga(
                    time + steps,
                    self.population,
                    None,
                    state)

            steps += 1

        return steps

    def _run_trial_exploit(self):
        pass

    def _collect_metrics(self, trials, max_steps):
        return {
            'population': len(self.population),
            'numerosity': sum(cl.num for cl in self.population),
            'reliable': len([cl for cl in
                             self.population if cl.is_reliable()]),
            'fitness': (sum(cl.fitness for cl in self.population) /
                        len(self.population)),
            'trials': trials,
            'max_steps': max_steps
        }
