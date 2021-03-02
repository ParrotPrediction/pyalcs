from lcs.agents.macs.macs import Classifier, Configuration

import pytest


class TestClassifier:

    @pytest.fixture
    def cfg(self):
        return Configuration(4, 2)

    @pytest.mark.parametrize('_g, _b, _er, _res', [
        (0, 0, 5, False),
        (0, 2, 2, True),
        (1, 2, 2, False),
    ])
    def test_should_detect_inaccurate(self, _g, _b, _er, _res, cfg):
        # given
        cl = Classifier(cfg=cfg)
        cl.g = _g
        cl.b = _b
        cfg.er = _er

        # then
        assert cl.is_inaccurate == _res
