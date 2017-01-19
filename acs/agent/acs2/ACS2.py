import logging

from acs.agent.Agent import Agent
from acs.environment.Environment import Environment
from .ALP import apply_alp
from .GA import apply_ga
from .RL import apply_rl
from .ACS2Utils import generate_initial_classifiers,\
    generate_match_set, generate_action_set, choose_action

logger = logging.getLogger(__name__)


class ACS2(Agent):

    def __init__(self, environment: Environment):
        super().__init__(environment)

        self.time = 0
        self.classifiers = []
        self.match_set = []
        self.action_set = []
        self.perception = None
        self.action = None
        self.reward = None
        self.previous_action_set = None
        self.previous_perception = None

    def evaluate(self, generations, **kwargs) -> None:
        for _ in range(generations):
            logger.info('\n\nGeneration [%d]', self.time)

            # Reset the environment and put the animat randomly
            # inside the maze when we are starting the simulation
            # or when he found the reward (next trial)
            if self.time == 0 or self.env.animat_has_finished():
                self.env.reset_animat_state()
                self.env.insert_animat()

            # Generate initial (general) classifiers when the simulation
            # Just starts or there is none classifier in the population
            if self.time == 0 or len(self.classifiers) == 0:
                self.classifiers = generate_initial_classifiers()

            # Get the animat perception
            self.perception = list(self.env.get_animat_perception())

            # Select classifiers matching the perception
            self.match_set = generate_match_set(self.classifiers,
                                                self.perception)

            if self.previous_action_set is not None:
                apply_alp(self.classifiers,
                          self.action,
                          self.time,
                          self.action_set,
                          self.perception,
                          self.previous_perception)
                apply_rl(self.match_set,
                         self.action_set,
                         self.reward)
                apply_ga(self.classifiers,
                         self.action_set,
                         self.time)

            # Remove previous action set
            self.previous_action_set = None

            self.action = choose_action(self.match_set)
            self.action_set = generate_action_set(self.match_set, self.action)

            # Execute action and obtain reward
            self.reward = self.env.execute_action(self.action)

            # Next time slot
            self.time += 1
            self.previous_perception = self.perception
            self.previous_action_set = self.action_set

            self.perception = self.env.get_animat_perception()

            if self.time % 100 == 0:
                logger.info('=== 100 ===')
                # Some debug / measurements here
