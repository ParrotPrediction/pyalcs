import logging
import random
from lcs import Perception
from lcs.agents.Agent import TrialMetrics
from . import ClassifiersList, Configuration, Classifier
from ...agents import Agent

logger = logging.getLogger(__name__)


class fACS2(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None) -> None:
        self.cfg = cfg
        self.population = population or ClassifiersList()

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, time, current_trial=None):

        logger.debug("** Running trial explore ** ")
        # Initial conditions
        steps = 0
        state = env.reset()
        action = env.action_space.sample()
        last_reward = 0
        prev_state = Perception.empty()
        action_set = ClassifiersList()
        done = False

        while not done:
            state = Perception(state)
            state_to_calculate = Perception(env.change_state_type(state))
            membership_func_values = env.to_membership_function(state)
            match_set = self.population.form_match_set(membership_func_values)

            if steps > 0:
                # Apply learning in the last action set
                ClassifiersList.apply_alp(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    action,
                    state_to_calculate,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma
                )
                if self.cfg.do_ga:
                    ClassifiersList.apply_ga(
                        time + steps,
                        self.population,
                        match_set,
                        action_set,
                        state_to_calculate,
                        self.cfg.theta_ga,
                        self.cfg.mu,
                        self.cfg.chi,
                        self.cfg.theta_as,
                        self.cfg.do_subsumption,
                        self.cfg.theta_exp)

            if random.random() > self.cfg.epsilon:
                action = self.select_action(env, match_set, membership_func_values)
            else:
                action = random.choice(range(self.cfg.number_of_possible_actions))

            action_set = match_set.form_action_set(action)

            prev_state = Perception(state_to_calculate)
            raw_state, last_reward, done, _ = env.step(action)

            state = Perception(raw_state)
            state_to_calculate = Perception(env.change_state_type(state))

            if done:
                ClassifiersList.apply_alp(
                    self.population,
                    ClassifiersList(),
                    action_set,
                    prev_state,
                    action,
                    state_to_calculate,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    0,
                    self.cfg.beta,
                    self.cfg.gamma)
                if self.cfg.do_ga:
                    ClassifiersList.apply_ga(
                        time + steps,
                        self.population,
                        ClassifiersList(),
                        action_set,
                        state_to_calculate,
                        self.cfg.theta_ga,
                        self.cfg.mu,
                        self.cfg.chi,
                        self.cfg.theta_as,
                        self.cfg.do_subsumption,
                        self.cfg.theta_exp)

            steps += 1

        return TrialMetrics(steps, last_reward)

    def _run_trial_exploit(self, env, time=None, current_trial=None) \
            -> TrialMetrics:

        logger.debug("** Running trial exploit **")
        # Initial conditions
        steps = 0

        state = Perception(env.reset())

        last_reward = 0
        action_set = ClassifiersList()
        done = False

        while not done:
            env.render()
            state = Perception(state)
            membership_func_values = env.to_membership_function(state)

            match_set = self.population.form_match_set(membership_func_values)

            if steps > 0:
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma)

            # Here when exploiting always choose best action
            action = self.select_action(env, match_set, membership_func_values)
            action_set = match_set.form_action_set(action)

            raw_state, last_reward, done, _ = env.step(action)

            state = Perception(raw_state)

            if done:
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.cfg.beta, self.cfg.gamma)

            steps += 1

        return TrialMetrics(steps, last_reward)

    def calculate_min_value_for_each_clasifier(self,
                                               match_set,
                                               memberships_values):
        """
        Select min value from all memberships function values
        where classifier had active rule.

        Parameters
        ----------
        match_set
            match set of classifiers
        memberships_values
            membership values for current environment state

        Returns
        -------
        [[float, int]]
            min values of membership for each classifier
            and action of that classifier

        """

        if not match_set:
            return
        elif type(match_set) == Classifier:
            conditions = [match_set.condition]
            actions = [match_set.action]
        else:
            conditions = [clf.condition for clf in match_set]
            actions = [clf.action for clf in match_set]
        values = []
        for conds, a in zip(conditions, actions):
            conditions_values = []
            for input_values in memberships_values:
                for c, m in zip(conds, input_values):
                    if c == self.cfg.classifier_wildcard:
                        continue
                    conditions_values.append(float(c) * m)
            if True in conditions_values:
                values.append((min(c for c in conditions_values if c > 0), a))
        return values

    def select_max_action_value(self, output_values):
        """
        Select max membership value for each possible action

        Parameters
        ----------
        output_values
            min values of each classifier and proposed action

        Returns
        -------
        possible_actions
            all posible actions with max membership value
            of it

        """
        possible_actions = [0 for _ in range(
            self.cfg.number_of_possible_actions)]
        for value, action_index in output_values:
            if possible_actions[action_index] < value:
                possible_actions[action_index] = value
        return possible_actions

    def select_action(self, env, match_set, memberships_values):
        """
        Select final action from match_set

        Parameters
        ----------
        env
        match_set
        memberships_values

        Returns
        -------
        action
            selected action

        """
        calculate = self.calculate_min_value_for_each_clasifier
        min_values = calculate(match_set, memberships_values)
        if not min_values:
            return random.choice(range(self.cfg.number_of_possible_actions))
        actions = self.select_max_action_value(min_values)
        actions_func_shape = env. \
            calculate_final_actions_func_shape(actions)
        return round(env.calculate_centroid(
            actions_func_shape)[0])
