from dataclasses import dataclass

import pytest

import lcs.strategies.reinforcement_learning as rl


@dataclass
class Classifier:
    r: float
    ir: float


class TestReinforcementLearning:

    @pytest.mark.parametrize("_r0, reward, _r1", [
        (0.5, 0, 0.5),
        (0.5, 1, 0.55),
        (0.5, 10, 1.45),
    ])
    def test_should_perform_bucket_brigade_update(self, _r0, reward, _r1):
        # given
        prev_cl = Classifier(_r0, None)
        cl = Classifier(0.5, None)

        # when
        rl.bucket_brigade_update(cl, prev_cl, reward)

        # then
        assert cl.r == 0.5
        assert prev_cl.r == _r1
        assert cl.ir is None
        assert prev_cl.ir is None

    def test_should_perform_bucket_brigade_update_when_first_step(self):
        # given
        prev_cl = None
        cl = Classifier(0.5, None)

        # when
        rl.bucket_brigade_update(cl, prev_cl, 100)

        # then
        assert cl.r == 0.5
        assert prev_cl is None

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
