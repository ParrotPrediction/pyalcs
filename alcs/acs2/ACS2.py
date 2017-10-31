from alcs.acs2 import ClassifiersList

DO_GA = True


class ACS2:
    def __init__(self, population=None):
        if population is not None:
            self.population = population
        else:
            self.population = ClassifiersList()

    def explore(self, env, max_trials):
        current_trial = 0
        steps = 0

        metrics = []
        while current_trial < max_trials:
            steps_in_trial = self._run_trial_explore(env, steps)
            steps += steps_in_trial

            # Collect metrics of trial
            metrics.append(self._collect_metrics(
                current_trial, steps_in_trial))

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
            metrics.append(self._collect_metrics(
                current_trial, steps_in_trial))

            current_trial += 1

        return self.population, metrics

    def _run_trial_explore(self, env, time):
        # Initial conditions
        steps = 0
        state = env.reset()

        action = None
        reward = None
        prev_state = None
        action_set = ClassifiersList()
        done = False

        while not done:
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
                if DO_GA:
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
                if DO_GA:
                    action_set.apply_ga(
                        time + steps,
                        self.population,
                        None,
                        state)

            steps += 1

        return steps

    def _run_trial_exploit(self, env):
        # Initial conditions
        steps = 0
        state = env.reset()

        reward = None
        action_set = ClassifiersList()
        done = False

        while not done:
            match_set = ClassifiersList.form_match_set(
                self.population, state)

            if steps > 0:
                action_set.apply_reinforcement_learning(
                    reward,
                    match_set.get_maximum_fitness())

            action = match_set.choose_action(epsilon=0.0)
            action_set = ClassifiersList.form_action_set(match_set, action)

            state, reward, done, _ = env.step(action)

            if done:
                action_set.apply_reinforcement_learning(reward, 0)

            steps += 1

        return steps

    def _collect_metrics(self, trial, steps):
        return {
            'population': len(self.population),
            'numerosity': sum(cl.num for cl in self.population),
            'reliable': len([cl for cl in
                             self.population if cl.is_reliable()]),
            'fitness': (sum(cl.fitness for cl in self.population) /
                        len(self.population)),
            'trial': trial,
            'steps': steps
        }
