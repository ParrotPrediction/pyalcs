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

    def test_foo(self, mp):
        # given
        cfg = ACS2Configuration(mp.env.observation_space.n, 2, do_ga=True)

        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore_exploit(mp, 50)

        # then
        reliable = [cl for cl in population if cl.is_reliable()]
        print(len(population))
