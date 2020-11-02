import logging

import numpy as np

from lcs import Perception
from lcs.agents.Agent import TrialMetrics
from lcs.agents.acs2 import ClassifiersList
from lcs.strategies.action_selection import BestAction, RandomAction
from . import Configuration
from ...agents import Agent

logger = logging.getLogger(__name__)


class AACS2(Agent):

    def __init__(self,
                 cfg: Configuration,
                 estimated_average_reward: float = 0,
                 population: ClassifiersList = None) -> None:
        self.cfg = cfg
        self.rho = estimated_average_reward
        self.population = population or ClassifiersList()

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, time, current_trial=None) \
        -> TrialMetrics:

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

        prev_M_best_fitness = 0
        was_greedy = False

        while not done:
            state = Perception(state)
            match_set = self.population.form_match_set(state)

            if steps > 0:
                # Apply learning in the last action set
                ClassifiersList.apply_alp(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                self.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    prev_M_best_fitness,
                    match_set.get_maximum_fitness(),
                    was_greedy)
                if self.cfg.do_ga:
                    ClassifiersList.apply_ga(
                        time + steps,
                        self.population,
                        match_set,
                        action_set,
                        state,
                        self.cfg.theta_ga,
                        self.cfg.mu,
                        self.cfg.chi,
                        self.cfg.theta_as,
                        self.cfg.do_subsumption,
                        self.cfg.theta_exp)

            action, was_greedy = self._epsilon_greedy(match_set)
            iaction = self.cfg.environment_adapter.to_lcs_action(action)
            logger.debug("\tExecuting action: [%d]", action)
            action_set = match_set.form_action_set(action)

            prev_state = Perception(state)
            prev_M_best_fitness = match_set.get_maximum_fitness()

            raw_state, last_reward, done, _ = env.step(iaction)

            state = self.cfg.environment_adapter.to_genotype(raw_state)
            state = Perception(state)

            if done:
                ClassifiersList.apply_alp(
                    self.population,
                    ClassifiersList(),
                    action_set,
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                self.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    prev_M_best_fitness,
                    0,
                    was_greedy)
                if self.cfg.do_ga:
                    ClassifiersList.apply_ga(
                        time + steps,
                        self.population,
                        ClassifiersList(),
                        action_set,
                        state,
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

        prev_M_best_fitness = 0

        while not done:
            match_set = self.population.form_match_set(state)

            if steps > 0:
                self.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    prev_M_best_fitness,
                    match_set.get_maximum_fitness(),
                    False)

            # Here when exploiting always choose best action
            action = BestAction(
                all_actions=self.cfg.number_of_possible_actions)(match_set)
            iaction = self.cfg.environment_adapter.to_env_action(action)
            action_set = match_set.form_action_set(action)

            prev_M_best_fitness = match_set.get_maximum_fitness()

            raw_state, last_reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(raw_state)
            state = Perception(state)

            if done:
                self.apply_reinforcement_learning(
                    action_set, last_reward, prev_M_best_fitness, 0, False)

            steps += 1

        return TrialMetrics(steps, last_reward)

    def apply_reinforcement_learning(self,
                                     action_set: ClassifiersList,
                                     reward: int,
                                     p0: float,  # [M]t-1 best fitness
                                     p1: float,  # [M] best fitness
                                     was_greedy: bool = False) -> None:

        if was_greedy:
            if self.cfg.rho_update_version == '1':
                self.rho += self.cfg.zeta * (reward - self.rho)
            elif self.cfg.rho_update_version == '2':
                self.rho += self.cfg.zeta * (reward + p0 - p1 - self.rho)
            else:
                raise ValueError('Invalid rho update version')

        R = reward - self.rho + p1

        for cl in action_set:
            cl.r += self.cfg.beta * (R - cl.r)
            cl.ir += self.cfg.beta * (reward - cl.ir)

    def _epsilon_greedy(self, match_set: ClassifiersList):
        # Epsilon greedy action selection returning tuple - action and
        # information whether it was best possible move
        all_actions = self.cfg.number_of_possible_actions

        if np.random.rand() < self.cfg.epsilon:
            return RandomAction(all_actions=all_actions)(match_set), False
        else:
            return BestAction(all_actions=all_actions)(match_set), True
