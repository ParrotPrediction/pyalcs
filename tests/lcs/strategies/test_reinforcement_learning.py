from dataclasses import dataclass

import pytest

import lcs.strategies.reinforcement_learning as rl


@dataclass
class Classifier:
    r: float
    ir: float


class TestReinforcementLearning:

    @pytest.mark.parametrize("_r0, _r1, _ir0, _ir1", [
        (0.5, 97.975, 0.0, 50.0)
    ])
    def test_should_update_classifier(self, _r0, _r1, _ir0, _ir1):
        # given
        cl = Classifier(r=_r0, ir=_ir0)
        beta = 0.05
        gamma = 0.95
        env_reward = 1000
        max_match_set_fitness = 1000

        # when
        rl.update_classifier(cl,
                             env_reward,
                             max_match_set_fitness,
                             beta, gamma)

        # then
        assert abs(cl.r - _r1) < 0.001
        assert abs(cl.ir - _ir1) < 0.001
