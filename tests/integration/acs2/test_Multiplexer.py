import gym
# noinspection PyUnresolvedReferences
import gym_multiplexer
import pytest

from examples.acs2.boolean_multiplexer.utils import calculate_performance
from lcs.agents.acs2 import ACS2, Configuration, EnvironmentAdapter
from .utils import count_macroclassifiers, count_microclassifiers


class MultiplexerAdapter(EnvironmentAdapter):
    @staticmethod
    def env_state_to_acs(env_state):
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
                            environment_adapter=MultiplexerAdapter(),
                            performance_fcn=calculate_performance,
                            performance_fcn_params={'ctrl_bits': 2})
        agent = ACS2(cfg)

        # when
        population, metrics = agent.explore(mp, 10)

        # then
        for metric in metrics:
            assert metric['performance']['was_correct'] in {0, 1}
