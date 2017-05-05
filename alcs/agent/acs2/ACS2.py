import logging

from alcs.agent.Agent import Agent
from alcs.agent.acs2 import ClassifiersList
from alcs.environment.Environment import Environment
from alcs.agent.acs2 import Constants as c

logger = logging.getLogger(__name__)


class ACS2(Agent):

    def __init__(self):
        super().__init__()
        # TODO: Here agent will be initialize with constants

    def evaluate(self,
                 env: Environment,
                 experiments: int,
                 max_steps: int):

        self.metrics.clear()

        for experiment_id in range(0, experiments):
            self._run_experiment(env, experiment_id, max_steps)

        return self.metrics

    def _run_experiment(self,
                        env: Environment,
                        experiment_id: int,
                        max_steps: int) -> ClassifiersList:
        trial = 0
        all_steps = 0

        population = ClassifiersList()

        while all_steps < max_steps:
            logger.info("Trial/steps: [{}/{}]".format(trial, all_steps))
            steps_in_trial = self._start_one_trial_explore(population, env, all_steps, max_steps)

            all_steps += steps_in_trial
            trial += 1

            logger.debug("Collecting metrics...")
            self.acquire_metrics(
                experiment=experiment_id,
                maze=env,
                steps=steps_in_trial,
                classifiers=population
            )

            logger.info("Know: {:.1f}% Pop: {} Num: {} Rel: {} Ina: {} Fit: {:.1f} Spec: {:.2f}\n".format(
                self.metrics['knowledge'][-1],
                len(population),
                sum(cl.num for cl in population),
                len([cl for cl in population if cl.is_reliable()]),
                len([cl for cl in population if cl.q < 0.1]),
                sum(cl.fitness for cl in population) / len(population),
                self.metrics['specificity'][-1]
            ))

        return population

    @staticmethod
    def _start_one_trial_explore(population: ClassifiersList,
                                 env: Environment,
                                 time: int,
                                 max_steps: int):
        steps = 0
        env.insert_animat()

        action = None
        reward = None
        previous_situation = None
        action_set = ClassifiersList()

        situation = env.get_animat_perception()

        while not env.trial_finished() and time + steps <= max_steps and steps < c.MAX_TRIAL_STEPS:
            match_set = ClassifiersList.form_match_set(population, situation)

            if steps > 0:
                # Apply learning in the last action set
                action_set.apply_alp(previous_situation, action, situation,
                                     time + steps, population, match_set)
                action_set.apply_reinforcement_learning(reward,
                                                        match_set.get_maximum_fitness())
                action_set.apply_ga(time + steps, population, match_set,
                                    situation)

            action = match_set.choose_action(epsilon=c.EPSILON)
            action_set = ClassifiersList.form_action_set(match_set, action)

            reward = env.execute_action(action)

            previous_situation = situation
            situation = env.get_animat_perception()

            if env.trial_finished():
                action_set.apply_alp(previous_situation, action, situation,
                                     time + steps, population, None)
                action_set.apply_reinforcement_learning(reward, 0)
                action_set.apply_ga(time + steps, population, None, situation)

            steps += 1

        return steps
