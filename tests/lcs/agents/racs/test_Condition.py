import pytest
from lcs.agents.racs import Configuration, Condition


class TestCondition:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder_bits=4)

    def test_should_create_generic_condition(self, cfg):
        # when
        cond = Condition.generic(cfg)

        # then
        assert len(cond) == cfg.classifier_length
        for allele in cond:
            assert allele == cfg.classifier_wildcard
