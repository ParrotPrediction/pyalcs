import pytest
from lcs.agents.racs import Configuration, Effect


class TestEffect:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder_bits=4)

    def test_should_create_pass_through_effect(self, cfg):
        # when
        effect = Effect.pass_through(cfg)

        # then
        assert len(effect) == cfg.classifier_length
        for allele in effect:
            assert allele == cfg.classifier_wildcard
