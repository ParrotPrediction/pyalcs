# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for running OpenAI Gym envs"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import gin
from dopamine.common import iteration_statistics
from dopamine.common import logger


def load_gin_configs(gin_files, gin_bindings):
    """Loads gin configuration files.

    Args:
      gin_files: list, of paths to the gin configuration files for this
        experiment.
      gin_bindings: list, of gin parameter bindings to override the values in
        the config files.
    """
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False)


@gin.configurable
class Runner(object):
    """Object that handles running OpenAI Gym experiments.

    Here we use the term 'experiment' to mean simulating interactions between the
    agent and the environment and reporting some statistics pertaining to these
    interactions.

    A simple scenario to train a DQN agent is as follows:

    ```python
    base_dir = '/tmp/simple_example'
    def create_agent(sess, environment):
      return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
    runner = Runner(base_dir, create_agent, game_name='Pong')
    runner.run()
    ```
    """

    def __init__(self,
                 base_dir,
                 create_agent_fn,
                 environment=None,
                 logging_file_prefix='log',
                 log_every_n=1,
                 num_iterations=200,
                 training_steps=1,
                 evaluation_steps=1,
                 max_steps_per_episode=27000):
        """Initialize the Runner object in charge of running a full experiment.

        Args:
          base_dir: str, the base directory to host all required sub-directories.
          create_agent_fn: A function that takes as args a Tensorflow session and an
            OpenAI Gym environment, and returns an agent.
          environment: OpenAI environment (required).
          logging_file_prefix: str, prefix to use for the log files.
          log_every_n: int, the frequency for writing logs.
          num_iterations: int, the iteration number threshold (must be greater than
            start_iteration).
          training_steps: int, the number of training steps to perform.
          evaluation_steps: int, the number of evaluation steps to perform.
          max_steps_per_episode: int, maximum number of steps after which an episode
            terminates.

        This constructor will take the following actions:
        - Initialize an environment.
        - Initialize a logger.
        - Initialize an agent.
        """
        assert base_dir and environment is not None
        self._logging_file_prefix = logging_file_prefix
        self._log_every_n = log_every_n
        self._num_iterations = num_iterations
        self._training_steps = training_steps
        self._evaluation_steps = evaluation_steps
        self._max_steps_per_episode = max_steps_per_episode
        self._base_dir = base_dir
        self._create_directories()

        self._start_iteration = 0
        self._environment = environment
        self._agent = create_agent_fn(self._environment)

    def _create_directories(self):
        """Create necessary sub-directories."""
        self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))

    def _initialize_episode(self):
        """Initialization for a new episode.

        Returns:
          action: int, the initial action chosen by the agent.
        """
        initial_observation = self._environment.reset()
        return self._agent.begin_episode(initial_observation)

    def _run_one_step(self, action):
        """Executes a single step in the environment.

        Args:
          action: int, the action to perform in the environment.

        Returns:
          The observation, reward, and is_terminal values returned from the
            environment.
        """
        observation, reward, is_terminal, _ = self._environment.step(action)
        return observation, reward, is_terminal

    def _end_episode(self, reward):
        """Finalizes an episode run.

        Args:
          reward: float, the last reward from the environment.
        """
        self._agent.end_episode(reward)

    def _run_one_episode(self):
        """Executes a full trajectory of the agent
        interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """
        step_number = 0
        total_reward = 0.

        action = self._initialize_episode()

        # Keep interacting until we reach a terminal state.
        while True:
            observation, reward, is_terminal = self._run_one_step(action)

            total_reward += reward
            step_number += 1

            if step_number == self._max_steps_per_episode:
                # Stop the run loop once we reach the true end of episode.
                break
            elif is_terminal:
                # If we lose a life but the episode is not over,
                # signal an artificial end of episode to the agent.
                break

        self._end_episode(reward)

        return step_number, total_reward

    def _run_one_phase(self, min_steps, statistics, run_mode_str):
        """Runs the agent/environment loop until a desired number of steps.

        We follow the Machado et al., 2017 convention of running full episodes,
        and terminating once we've run a minimum number of steps.

        Args:
          min_steps: int, minimum number of steps to generate in this phase.
          statistics: `IterationStatistics` object which records the experimental
            results.
          run_mode_str: str, describes the run mode for this agent.

        Returns:
          Tuple containing the number of steps taken in this phase (int), the sum of
            returns (float), and the number of episodes performed (int).
        """
        step_count = 0
        num_episodes = 0
        sum_returns = 0.

        while step_count < min_steps:
            steps, episode_return = self._run_one_episode()

            statistics.append({
                '{}_episode_lengths'.format(run_mode_str): steps,
                '{}_episode_returns'.format(run_mode_str): episode_return
            })
            step_count += steps
            sum_returns += episode_return
            num_episodes += 1

            sys.stdout.write('Steps executed: {} '.format(step_count) +
                             'Return: {}\r'.format(episode_return))
            sys.stdout.flush()

        return step_count, sum_returns, num_episodes

    def _run_train_phase(self, statistics):
        """Run training phase.

        Args:
          statistics: `IterationStatistics` object which records
          the experimental results. Note - This object is modified
          by this method.

        Returns:
          num_episodes: int, The number of episodes run in this phase.
          average_reward: The average reward generated in this phase.
        """
        # Perform the training phase, during which the agent learns.
        self._agent.eval_mode = False
        number_steps, sum_returns, num_episodes = self._run_one_phase(
            self._training_steps, statistics, 'train')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        statistics.append({'train_average_return': average_return})
        return num_episodes, average_return

    def _run_eval_phase(self, statistics):
        """Run evaluation phase.

        Args:
          statistics: `IterationStatistics` object which records the experimental
            results. Note - This object is modified by this method.

        Returns:
          num_episodes: int, The number of episodes run in this phase.
          average_reward: float, The average reward generated in this phase.
        """
        # Perform the evaluation phase -- no learning.
        self._agent.eval_mode = True
        _, sum_returns, num_episodes = self._run_one_phase(
            self._evaluation_steps, statistics, 'eval')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        statistics.append({'eval_average_return': average_return})
        return num_episodes, average_return

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction.

        An iteration involves running several episodes until a certain number of
        steps are obtained. The interleaving of train/eval phases implemented here
        are to match the implementation of (Mnih et al., 2015).

        Args:
          iteration: int, current iteration number

        Returns:
          A dict containing summary statistics for this iteration.
        """
        statistics = iteration_statistics.IterationStatistics()
        print('Starting iteration {}'.format(iteration))

        num_episodes_train, average_reward_train = self._run_train_phase(
            statistics)
        num_episodes_eval, average_reward_eval = self._run_eval_phase(
            statistics)

        # TODO do something with that

        return statistics.data_lists

    def _log_experiment(self, iteration, statistics):
        """Records the results of the current iteration.

        Args:
          iteration: int, iteration number.
          statistics: `IterationStatistics` object containing statistics to log.
        """
        self._logger['iteration_{:d}'.format(iteration)] = statistics
        if iteration % self._log_every_n == 0:
            self._logger.log_to_file(self._logging_file_prefix, iteration)

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        print('Beginning training...')

        if self._num_iterations <= self._start_iteration:
            print('num_iterations (%d) < start_iteration(%d)',
                               self._num_iterations, self._start_iteration)
            return

        for iteration in range(self._start_iteration, self._num_iterations):
            statistics = self._run_one_iteration(iteration)
            self._log_experiment(iteration, statistics)


@gin.configurable
class TrainRunner(Runner):
    """Object that handles running Atari 2600 experiments.

    The `TrainRunner` differs from the base `Runner` class in that it does not
    the evaluation phase. Logging for the train phase is preserved as before.
    """

    def __init__(self, base_dir, create_agent_fn):
        """Initialize the TrainRunner object in charge of running a full experiment.

        Args:
          base_dir: str, the base directory to host all required sub-directories.
          create_agent_fn: A function that takes as args a Tensorflow session and an
            Atari 2600 Gym environment, and returns an agent.
        """
        print('Creating TrainRunner ...')
        super(TrainRunner, self).__init__(
            base_dir=base_dir, create_agent_fn=create_agent_fn)
        self._agent.eval_mode = False

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction.

        An iteration involves running several episodes until a certain number of
        steps are obtained. This method differs from the `_run_one_iteration` method
        in the base `Runner` class in that it only runs the train phase.

        Args:
          iteration: int, current iteration number, used as a global_step for saving
            Tensorboard summaries.

        Returns:
          A dict containing summary statistics for this iteration.
        """
        statistics = iteration_statistics.IterationStatistics()

        num_episodes_train, average_reward_train = self._run_train_phase(
            statistics)


        return statistics.data_lists

