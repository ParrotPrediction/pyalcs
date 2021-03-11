import pytest
import numpy as np

from lcs.agents.xcs import Configuration

# py.test -max_population 4 --cov=xcs tests/lcs/agents/xcs


class TestConfiguration:

    @pytest.fixture
    def cfg(self):
        return Configuration(number_of_actions=4)

    def test_minimum(self, cfg):
        assert float(np.finfo(np.float32).tiny) == cfg.initial_prediction


