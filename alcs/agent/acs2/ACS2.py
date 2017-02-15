import logging
from collections import defaultdict

from alcs.agent.Agent import Agent
from alcs.environment.Environment import Environment
from alcs.helpers.metrics import calculate_achieved_knowledge
from .ALP import apply_alp
from .GA import apply_ga
from .RL import apply_rl
from .ACS2Utils import generate_initial_classifiers,\
    generate_match_set, generate_action_set, choose_action

logger = logging.getLogger(__name__)


class ACS2(Agent):

    def __init__(self, environment: Environment,
                 epsilon: float = 0.4,
                 beta: float = 0.2,
                 gamma: float = 0.95):
        """
        :param environment: the environment in which agent will operate.
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
        """
        super().__init__(environment)
        self.classifiers = generate_initial_classifiers()

        # Algorithm constants
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma

    def evaluate(self, steps, **kwargs):
        """
        Evaluates ACS2 algorithm on given environment for certain number
        of generations

        :param steps: number of generations to run
        :param kwargs: additonal parameters (none at the moment)

        :return: final classifier list and metrics
        """
        performance_metrics = defaultdict(list)

        time = 0
        trial = 0
        perception = None
        action = None
        action_set = None
        reward = None
        previous_perception = None
        previous_action_set = None

        self.env.insert_animat()

        # Get the animat initial perception
        perception = self.env.get_animat_perception()

        while time < steps:
            logger.info('\n\nTrial/step [%d/%d]\t\t%s',
                        trial, time, perception)

            finished = False

            # Reset the environment and put the animat randomly
            # inside the maze when he found the reward (next trial starts)
            if self.env.animat_has_finished():
                finished = True
                trial += 1
                self.env.insert_animat()

            # Generate initial (general) classifiers if no classifier
            # are in the population.
            if len(self.classifiers) == 0:
                self.classifiers = generate_initial_classifiers()

            # Select classifiers matching the perception
            match_set = generate_match_set(self.classifiers, perception)

            # If not the beginning of the trial
            if previous_action_set is not None:  # time != 0
                logger.info("Triggering learning modules on previous "
                            "action set")
                apply_alp(self.classifiers,
                          action,
                          time,
                          previous_action_set,
                          perception,
                          previous_perception,
                          self.beta)
                apply_rl(match_set,
                         previous_action_set,
                         reward,
                         self.beta,
                         self.gamma)
                apply_ga(self.classifiers,
                         previous_action_set,
                         time)

            # Remove previous action set
            previous_action_set = None

            action = choose_action(match_set, self.epsilon)
            action_set = generate_action_set(match_set, action)

            # Execute action and obtain reward
            reward = self.env.execute_action(action)

            # Next time slot
            time += 1
            previous_perception = perception
            perception = self.env.get_animat_perception()

            # If new state was introduced
            if self.env.move_was_successful():
                logger.info("Move successful. Triggering learning modules")
                apply_alp(self.classifiers,
                          action,
                          time,
                          action_set,
                          perception,
                          previous_perception,
                          self.beta)
                apply_rl(match_set,
                         action_set,
                         reward,
                         self.beta,
                         self.gamma)
                apply_ga(self.classifiers,
                         action_set,
                         time)

            previous_action_set = action_set

            # Metrics for calculating performance
            cls_num = len(self.classifiers)
            s_fitness = sum(cl.fitness() for cl in self.classifiers)
            s_spec_pop = sum(cl.get_condition_specificity()
                             for cl in self.classifiers)
            knowledge = calculate_achieved_knowledge(
                self.env, self.classifiers)

            performance_metrics['time'].append(time)
            performance_metrics['found_reward'].append(finished)
            performance_metrics['total_classifiers'].append(cls_num)
            performance_metrics['spec_pop'].append(s_spec_pop / cls_num)
            performance_metrics['achieved_knowledge'].append(knowledge)
            performance_metrics['average_fitness'].append(s_fitness / cls_num)

        return self.classifiers, performance_metrics
