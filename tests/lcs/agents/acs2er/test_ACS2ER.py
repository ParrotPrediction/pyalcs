import pytest
from lcs.Perception import Perception
from lcs.agents.acs2er.ACS2ER import ACS2ER
from lcs.agents.acs2er.Configuration import Configuration
from lcs.agents.acs2er.ReplayMemorySample import ReplayMemorySample

from lcs.agents.acs2er.ReplayMemory import ReplayMemory

class TestReplayMemory:

    @pytest.fixture
    def cfg(self):
        return Configuration(    
            classifier_length=2,
            number_of_possible_actions=2,
            er_buffer_size=100,
            er_min_samples=20,
            er_samples_number=2)
            
    @pytest.fixture
    def rm(self):
        rm: ReplayMemory = ReplayMemory(max_size=100)
        sample: ReplayMemorySample = ReplayMemorySample(Perception(['0', '0']), 1, 0.5, Perception(['1', '0']), False)
        rm.update(sample)
        return rm

    def test_explore_10_trials_singlestep_20_min_samples_learning_not_begins(self, cfg, rm):
        # Arrange
        steps = []
        env = EnvMock(steps, 1)
        agent = ACS2ER(cfg, replay_memory=rm)
    
        # Act
        _ = agent.explore(env, 10)

        # Assert
        assert len(agent.get_population()) == 0
        assert len(rm) == 10 + 1

    def test_explore_50_trials_threestepspertrial_20_min_samples_learning_begins(self, cfg, rm):
        # Arrange
        steps = []
        env = EnvMock(steps, 3)
        agent = ACS2ER(cfg, replay_memory=rm)
    
        # Act
        _ = agent.explore(env, 50)

        # Assert
        assert len(agent.get_population()) > 0
        assert len(rm) == 100

class EnvMock:
    def __init__(self, steps, trial_length):
        self.steps = steps
        self.trial_length = trial_length
        self.trial_steps_count = 0
        self.action_space = ActionSpaceMock()

    def reset(self):
        return ['0', '0']

    def step(self, action):
        self.steps.append(action)
        self.trial_steps_count += 1

        if(self.trial_steps_count >= self.trial_length):
            self.trial_steps_count = 0
            return ['1', '1'], 10, True, None

        return ['1', '0'], 1, False, None

class ActionSpaceMock:
    def sample(self):
        return 1