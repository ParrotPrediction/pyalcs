import logging

from alcs.agent.Agent import Agent
from alcs.environment.Environment import Environment
from alcs.strategies.ActionSelection import ActionSelection, Random
from .ALP import apply_alp
from .GA import apply_ga
from .RL import apply_rl
from .ACS2Utils import generate_initial_classifiers,\
    generate_match_set, generate_action_set, choose_action

logger = logging.getLogger(__name__)


class ACS2(Agent):
    def __init__(self,
                 epsilon: float = 0.5,
                 beta: float = 0.2,
                 gamma: float = 0.95,
                 mu: float = 0.3,
                 x: float = 0.8,
                 strategy: ActionSelection = Random()):
        """
        :param epsilon: The 'exploration probability' [0-1]. Specifies the
        probability of choosing a random action. The fastest model learning is
        usually achieved by pure random exploration.
        :param beta: The 'learning rate' - used in ALP and RL. Updates
        affecting q, r, ir, aav. parameters approach an approximation of
        their actual value but the more noisy the approximation is.
        :param gamma: The 'discount factor' [0-1] determines the reward
        distribution over the environmental model. It essentially specifies
        to what extend future reinforcement influences current behaviour.
        The closer to 1, the more influence delayed reward has on current
        behaviour.
        :param mu: The 'mutation rate' [0-1] specifies the probability of
        changing a specified attribute in the conditions of an offspring
        to a #-symbol in a GA. Default to 0.3. Lower values decrease the
        generalization pressure and consequently decrease the speed of
        conversion in the population. Higher values on the other hand can
        also decrease conversion because of the higher amount of over-general
        classifiers.
        :param x: # The 'crossover probability' [0-1] specifies the
        probability of applying crossover in the conditions of the offspring
        when a GA is applied. Default to 0.8. It seems to influence the
        process only slightly. No problem was found so far in which crossover
        actually has a significant effect.
        :param strategy: Exploration strategy. Default - random action
        """
        super().__init__()

        # Algorithm constants
        self.exploration_probability = epsilon
        self.learning_rate = beta
        self.discount_factor = gamma
        self.mutation_rate = mu
        self.crossover_probability = x
        self.exploration_strategy = strategy

    def evaluate(self,
                 environment: Environment,
                 steps: int,
                 max_steps: int = None):
        """
        Evaluates ACS2 algorithm on given environment for certain number
        of generations

        :param environment: environment to operate on
        :param steps: number of generations to run
        :param max_steps: maximum number of steps in trial (before resetting
        the agent)

        :return: final classifier list and metrics
        """
        self.metrics.clear()

        classifiers = []

        step = 0
        steps_in_trial = 0
        trial = 0
        perception = None
        action = None
        action_set = None
        reward = None
        previous_perception = None
        previous_action_set = None

        environment.insert_animat()

        # Get the animat initial perception
        perception = environment.get_animat_perception()

        while step < steps:
            logger.info('\n\nTrial/step [%d/%d]\t\t%s',
                        trial, step, perception)

            finished = False

            # Generate initial (general) classifiers if no classifier
            # are in the population.
            if len(classifiers) == 0:
                classifiers = generate_initial_classifiers()

            # Reset the environment and put the animat randomly
            # inside the maze when he found the reward (next trial starts)
            if environment.animat_has_finished():
                finished = True
                trial += 1
                steps_in_trial = 0
                environment.insert_animat()
                perception = environment.get_animat_perception()
                previous_action_set = None

            # If the animat will make more than `max_steps` in a single
            # trial put animal randomly somewhere else.
            if max_steps is not None and steps_in_trial == max_steps:
                logger.debug('Max steps in trial reached')
                steps_in_trial = 0
                environment.insert_animat()
                perception = environment.get_animat_perception()
                previous_action_set = None

            # Select classifiers matching the perception
            match_set = generate_match_set(classifiers, perception)

            # If not the beginning of the trial
            if previous_action_set is not None:  # time != 0
                logger.info("")
                logger.info("== Triggering learning modules on previous "
                            "action set ==")
                apply_alp(classifiers,
                          action,
                          step,
                          previous_action_set,
                          perception,
                          previous_perception,
                          self.learning_rate)
                apply_rl(match_set,
                         previous_action_set,
                         reward,
                         self.learning_rate,
                         self.discount_factor)
                apply_ga(classifiers,
                         previous_action_set,
                         step,
                         self.mutation_rate,
                         self.crossover_probability)

            # Remove previous action set
            previous_action_set = None

            action = choose_action(
                match_set,
                self.exploration_probability,
                self.exploration_strategy)
            action_set = generate_action_set(match_set, action)

            # Execute action and obtain reward
            logger.info("")
            reward = environment.execute_action(action)
            logger.info("Reward to distribute: %d", reward)

            # Next time slot
            step += 1
            steps_in_trial += 1

            logger.debug("Next trial started...")

            previous_perception = perception
            perception = environment.get_animat_perception()

            # If new state was introduced
            if environment.animat_has_finished():
                logger.info("")
                logger.info("== Move successful."
                            "Triggering learning modules ==")
                apply_alp(classifiers,
                          action,
                          step,
                          action_set,
                          perception,
                          previous_perception,
                          self.learning_rate)
                apply_rl(match_set,
                         action_set,
                         reward,
                         self.learning_rate,
                         self.discount_factor)
                apply_ga(classifiers,
                         action_set,
                         step,
                         self.mutation_rate,
                         self.crossover_probability)

            previous_action_set = action_set

            # Define variables for collecting metrics
            logger.debug("Collecting metrics...")
            self.acquire_metrics(
                step=step,
                maze=environment,
                classifiers=classifiers,
                was_successful=finished
            )

        return classifiers, self.metrics
