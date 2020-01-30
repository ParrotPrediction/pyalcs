import logging
import random

from lcs import Perception
from lcs.agents import Agent
from lcs.agents.Agent import TrialMetrics
from lcs.agents.acs import ClassifiersList, Configuration, Classifier

import lcs.agents.acs.alp as alp
import lcs.strategies.reinforcement_learning as rl

logger = logging.getLogger(__name__)


class ACS(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None) -> None:
        self.cfg = cfg
        self.population = population or self._initial_population()

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        logger.debug("** Running trial explore ** ")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)
        action = env.action_space.sample()
        last_reward = 0
        prev_state = Perception.empty()
        selected_cl = None
        prev_selected_cl = None
        done = False

        while not done:
            state = Perception(state)
            match_set = self.population.form_match_set(state)

            if steps > 0:
                alp.apply(prev_state,
                          state,
                          selected_cl,
                          self.population)
                rl.bucket_brigade_update(
                    selected_cl,
                    prev_selected_cl,
                    last_reward)

            prev_selected_cl = selected_cl

            # TODO: you can do it better
            if random.random() < self.cfg.epsilon:
                selected_cl = random.choice(match_set)
            else:
                selected_cl = self._best_cl(match_set)

            action = selected_cl.action
            iaction = self.cfg.environment_adapter.to_lcs_action(action)
            logger.debug("\tExecuting action: [%d]", action)

            prev_state = Perception(state)

            raw_state, last_reward, done, _ = env.step(iaction)

            state = self.cfg.environment_adapter.to_genotype(raw_state)
            state = Perception(state)

            if done:
                alp.apply(prev_state,
                          state,
                          selected_cl,
                          self.population)
                rl.bucket_brigade_update(
                    selected_cl,
                    prev_selected_cl,
                    last_reward)


            steps += 1

        return TrialMetrics(steps, last_reward)

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        logger.debug("** Running trial exploit **")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)
        action = env.action_space.sample()
        last_reward = 0
        prev_state = Perception.empty()
        selected_cl = None
        prev_selected_cl = None
        done = False

        while not done:
            state = Perception(state)
            match_set = self.population.form_match_set(state)

            selected_cl = self._best_cl(match_set)
            action = selected_cl.action
            iaction = self.cfg.environment_adapter.to_lcs_action(action)
            logger.debug("\tExecuting action: [%d]", action)

            raw_state, last_reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(raw_state)
            state = Perception(state)
            steps += 1

        return TrialMetrics(steps, last_reward)

    def _initial_population(self):
        cls = []
        for action in range(0, self.cfg.number_of_possible_actions):
            cls.append(Classifier.general(action, cfg=self.cfg))

        return ClassifiersList(*cls)


    def _best_cl(self, match_set):
        return max(match_set, key=lambda cl: cl.fitness)
