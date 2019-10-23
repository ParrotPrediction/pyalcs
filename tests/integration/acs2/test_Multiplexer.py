import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer
import pytest

from lcs.agents import EnvironmentAdapter
from lcs.agents.acs2 import ACS2, Configuration
from .utils import count_macroclassifiers, count_microclassifiers


class MultiplexerAdapter(EnvironmentAdapter):
    @classmethod
    def to_genotype(cls, env_state):
        return [str(x) for x in env_state]


class TestMultiplexer:

    @pytest.fixture
    def mp(self):
        return gym.make('boolean-multiplexer-6bit-v0')

    def test_should_initialize_multiplexer_environment(self, mp):
        assert mp is not None

    def test_should_be_no_duplicated_classifiers_without_ga(self, mp):
        # given
        cfg = Configuration(mp.env.observation_space.n, 2,
                            environment_adapter=MultiplexerAdapter(),
                            do_ga=False)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(mp, 10)

        # then
        assert count_macroclassifiers(population) == len(set(population))

    def test_should_be_no_duplicated_classifiers_with_ga(self, mp):
        # given
        cfg = Configuration(mp.env.observation_space.n, 2,
                            environment_adapter=MultiplexerAdapter(),
                            do_ga=True)
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(mp, 1000)

        # then
        assert count_macroclassifiers(population) == len(set(population))
        assert count_microclassifiers(population) \
            > count_macroclassifiers(population)

    def test_should_evaluate_knowledge(self, mp):
        # given
        cfg = Configuration(mp.env.observation_space.n, 2,
                            do_ga=False,
                            environment_adapter=MultiplexerAdapter())
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(mp, 10)

        # then
        for metric in metrics:
            assert metric['reward'] in {0, 1000}
