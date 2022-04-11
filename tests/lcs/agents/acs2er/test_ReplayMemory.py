from lcs.Perception import Perception

from lcs.agents.acs2er.ReplayMemory import ReplayMemory
from lcs.agents.acs2er.ReplayMemorySample import ReplayMemorySample

class TestReplayMemory:

    def test_update_should_add_sample_properly(self):
        # Arrange
        rm: ReplayMemory = ReplayMemory(max_size=5)
        sample: ReplayMemorySample = ReplayMemorySample(Perception(['0']), 1, 0.5, Perception(['1']), False)

        # Act
        rm.update(sample)

        # Assert
        assert len(rm) == 1

    def test_update_should_drop_oldest_experience_if_size_exceeded(self):
        # Arrange
        rm: ReplayMemory = ReplayMemory(max_size=3)
        sample1: ReplayMemorySample = ReplayMemorySample(Perception(['0']), 1, 0.5, Perception(['1']), False)
        sample2: ReplayMemorySample = ReplayMemorySample(Perception(['0']), 2, 0.5, Perception(['1']), False)
        sample3: ReplayMemorySample = ReplayMemorySample(Perception(['0']), 3, 0.5, Perception(['1']), False)
        sample4: ReplayMemorySample = ReplayMemorySample(Perception(['0']), 4, 0.5, Perception(['1']), False)

        # Act
        rm.update(sample1)
        rm.update(sample2)
        rm.update(sample3)
        rm.update(sample4)

        # Assert
        assert len(rm) == 3
        assert rm[0].action == 2
        assert rm[1].action == 3
        assert rm[2].action == 4
