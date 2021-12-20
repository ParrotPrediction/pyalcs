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
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)
        action = env.action_space.sample()
        last_reward = 0
        prev_state = Perception.empty()
        action_set = ClassifiersList()
        done = False

        while not done:
            state = Perception(state)
            state_to_calculate = Perception(self.cfg.environment_adapter.change_state_type(state))
            membership_func_values = self.cfg.environment_adapter.to_membership_function(state)
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
                action = self.select_action(match_set, membership_func_values)
            else:
                action = random.choice(range(self.cfg.number_of_possible_actions))

            iaction = self.cfg.environment_adapter.to_lcs_action(action)
            action_set = match_set.form_action_set(action)

            prev_state = Perception(state_to_calculate)
            raw_state, last_reward, done, _ = env.step(iaction)

            state = self.cfg.environment_adapter.to_genotype(raw_state)
            state = Perception(state)
            state_to_calculate = Perception(self.cfg.environment_adapter.change_state_type(state))

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
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)
        state = Perception(state)

        last_reward = 0
        action_set = ClassifiersList()
        done = False

        while not done:
            env.render()
            state = Perception(state)
            membership_func_values = self.cfg.environment_adapter.to_membership_function(state)

            match_set = self.population.form_match_set(membership_func_values)

            if steps > 0:
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma)

            # Here when exploiting always choose best action
            action = self.select_action(match_set, membership_func_values)
            iaction = self.cfg.environment_adapter.to_env_action(action)
            action_set = match_set.form_action_set(action)

            raw_state, last_reward, done, _ = env.step(iaction)

            state = self.cfg.environment_adapter.to_genotype(raw_state)
            state = Perception(state)

            if done:
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.cfg.beta, self.cfg.gamma)

            steps += 1

        return TrialMetrics(steps, last_reward)

    def calculate_min_value_for_each_clasifier(self,
                                               match_set,
                                               memberships_values):
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
        possible_actions = [0 for _ in range(
            self.cfg.number_of_possible_actions)]
        for value, action_index in output_values:
            if possible_actions[action_index] < value:
                possible_actions[action_index] = value
        return possible_actions

    def select_action(self, match_set, memberships_values):
        calculate = self.calculate_min_value_for_each_clasifier
        min_values = calculate(match_set, memberships_values)
        if not min_values:
            return random.choice(range(self.cfg.number_of_possible_actions))
        actions = self.select_max_action_value(min_values)
        actions_func_shape = self.cfg.environment_adapter. \
            calculate_final_actions_func_shape(actions)
        return round(self.cfg.environment_adapter.calculate_centroid(
            actions_func_shape)[0])
