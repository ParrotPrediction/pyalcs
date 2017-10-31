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
            metrics.append(self._collect_metrics(env,
                                                 current_trial, steps_in_trial, steps))

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
            metrics.append(self._collect_metrics(env,
                                                 current_trial, steps_in_trial, steps))

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
            print("population size: %d, numerosity=%d" % (len(self.population), self.population.overall_numerosity()))
            match_set = ClassifiersList.form_match_set(self.population, state)
            # print("match_set size: %d" % len(match_set))

            if steps > 0:
                # Apply learning in the last action set
                action_set.apply_alp(
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.population,
                    match_set)
                # print("population before RL: size: %d, numerosity=%d" % (len(self.population),
                # self.population.overall_numerosity()))
                action_set.apply_reinforcement_learning(
                    reward,
                    match_set.get_maximum_fitness())
                # print("population after RL: size: %d, numerosity=%d" % (len(self.population),
                # self.population.overall_numerosity()))
                if DO_GA:
                    prev_num = self.population.overall_numerosity()
                    action_set.apply_ga(
                        time + steps,
                        self.population,
                        match_set,
                        state)
                    post_num = self.population.overall_numerosity()
                    # print("population after GA: size: %d, numerosity=%d" % (len(self.population), '
                    # self.population.overall_numerosity()))
                    # print("GA delta in numerosity: %d" % (post_num-prev_num))

            action = match_set.choose_action(epsilon=1.0)
            action_set = ClassifiersList.form_action_set(match_set, action)
            # print("action set size: %d, numerosity: %d" % (len(action_set), action_set.overall_numerosity()))

            prev_state = state
            state, reward, done, _ = env.step(action)

            if done:
                # print("population before alp2: size: %d, numerosity=%d" % (len(self.population),
                # self.population.overall_numerosity()))
                action_set.apply_alp(
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.population,
                    None)
                # print("population before RL2: size: %d, numerosity=%d" % (len(self.population),
                # self.population.overall_numerosity()))
                action_set.apply_reinforcement_learning(
                    reward,
                    0)
                # print("population after RL2: size: %d, numerosity=%d" % (len(self.population),
                # self.population.overall_numerosity()))
            if DO_GA:
                prev_num = self.population.overall_numerosity()
                action_set.apply_ga(
                    time + steps,
                    self.population,
                    None,
                    state)
                post_num = self.population.overall_numerosity()
                # print("population after GA2: size: %d, numerosity=%d" % (len(self.population),
                # self.population.overall_numerosity()))
                # print("GA2 delta in numerosity: %d" % (post_num-prev_num))

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

    def _collect_metrics(self, env, trial, steps, total_steps):
        return {
            'population': len(self.population),
            'numerosity': sum(cl.num for cl in self.population),
            'reliable': len([cl for cl in
                             self.population if cl.is_reliable()]),
            'fitness': (sum(cl.fitness for cl in self.population) /
                        len(self.population)),
            # 'knowledge': self.calculate_knowledge(env),
            'trial': trial,
            'steps': steps,
            'total_steps': total_steps
        }

    def calculate_knowledge(self, env):
        transitions = env.unwrapped.unwrapped.get_all_possible_transitions()

        # Take into consideration only reliable classifiers
        reliable_classifiers = [c for c in self.population if c.is_reliable()]

        # Count how many transitions are anticipated correctly
        nr_correct = 0

        # For all possible destinations from each path cell
        for start, action, end in transitions:
            p0 = env.maze.perception(*start)
            p1 = env.maze.perception(*end)

            if any([True for cl in reliable_classifiers if cl.predicts_successfully(p0, action, p1)]):
                nr_correct += 1

        return nr_correct / len(transitions) * 100.0
