import logging

from alcs.agent.Agent import Agent
from alcs.environment.Environment import Environment
from .ALP import apply_alp
from .GA import apply_ga
from .RL import apply_rl
from .ACS2Utils import generate_initial_classifiers,\
    generate_match_set, generate_action_set, choose_action, choose_best_action

logger = logging.getLogger(__name__)


class ACS2(Agent):
    def __init__(self,
                 epsilon: float = 0.5,
                 beta: float = 0.2,
                 gamma: float = 0.95,
                 mu: float = 0.3,
                 x: float = 0.8,
                 exploitation_mode=False):
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
        :param exploitation_mode: last half of steps will be used for
        exploitation only (selecting best action and no learning)
        """
        super().__init__()

        # Algorithm constants
        self.exploration_probability = epsilon
        self.learning_rate = beta
        self.discount_factor = gamma
        self.mutation_rate = mu
        self.crossover_probability = x
        self.exploration_mode = exploitation_mode

    def evaluate(self, environment: Environment, steps: int):
        """
        Evaluates ACS2 algorithm on given environment for certain number
        of generations

        :param environment: environment to operate on
        :param steps: number of steps to take

        :return: a tuple of final classifier list and obtained metrics
        """
        self.metrics.clear()

        step = 0
        trial = 0

        classifiers = []

        while step < steps:

            # Trial counters
            trial += 1
            steps_in_trial = 0

            # Initially place animat into environment
            environment.insert_animat()
            perception = environment.get_animat_perception()

            action = None
            reward = None
            previous_perception = None
            previous_action_set = None

            logger.info('\n\nTrial/step [%d/%d]\t\t%s',
                        trial, step, perception)

            # Each trial
            while not environment.trial_finished():

                if self.exploration_mode and step > steps / 2:
                    # Pure exploration
                    perception = environment.get_animat_perception()
                    match_set = generate_match_set(classifiers, perception)
                    action = choose_best_action(match_set)
                    environment.execute_action(action)

                    step += 1
                    steps_in_trial += 1

                    if steps_in_trial == 100:
                        # Infinite loop - not enough knowledge
                        exit(1)
                else:
                    # Normal learning mode

                    # Generate initial (general) classifiers if there are none
                    # in the current population.
                    if len(classifiers) == 0:
                        classifiers = generate_initial_classifiers()

                    # Select classifiers matching the perception
                    match_set = generate_match_set(classifiers, perception)

                    # If not the beginning of the experiment
                    if steps_in_trial != 0:
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

                    action = choose_action(match_set,
                                           self.exploration_probability)
                    action_set = generate_action_set(match_set, action)

                    # Execute action and obtain reward
                    reward = environment.execute_action(action)

                    step += 1
                    steps_in_trial += 1

                    previous_perception = perception
                    perception = environment.get_animat_perception()

                    # If animat has found the reward
                    if environment.trial_finished():
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

                # Define variables for collecting metrics each 10 steps
                logger.debug("Collecting metrics...")
                self.acquire_metrics(
                    step=step,
                    maze=environment,
                    classifiers=classifiers,
                    was_successful=environment.trial_finished()
                )

        return classifiers, self.metrics
