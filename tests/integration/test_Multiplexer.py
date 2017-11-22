import pytest
import gym

# noinspection PyUnresolvedReferences
import gym_multiplexer
from alcs.acs2 import ACS2Configuration
from alcs.acs2.ACS2 import ACS2


class TestMultiplexer:

    @pytest.fixture
    def mp(self):
        return gym.make('boolean-multiplexer-6bit-v0')

    def test_should_initialize_multiplexer_environment(self, mp):
        assert mp is not None

    def test_should_be_no_duplicated_classifiers_without_ga(self, mp):
        # given
        cfg = ACS2Configuration(mp.env.observation_space.n, 2, do_ga=False)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(mp, 1000)

        # then
        assert len(population) == len(set(population))

    # @pytest.mark.skip(reason="inserting in GA phase is bad")
    def test_should_be_no_duplicated_classifiers_with_ga(self, mp):
        # given
        cfg = ACS2Configuration(mp.env.observation_space.n, 2, do_ga=True)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(mp, 1000)

        # then
        assert len(population) == len(set(population))
        # maybe check numerosity
