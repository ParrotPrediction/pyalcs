from random import randint

import pytest

from lcs.agents.acs2 import Configuration, Classifier, ClassifiersList
from lcs.strategies.action_selection import exploit, choose_random_action, \
    choose_latest_action, choose_action_from_knowledge_array, choose_action


class TestActionSelection:

    @pytest.fixture
    def cfg(self):
        return Configuration(8, 8)

    def test_should_return_all_possible_actions(self, cfg):
        # given
        all_actions = cfg.number_of_possible_actions
        population = ClassifiersList()
        actions = set()

        # when
        for _ in range(1000):
            act = choose_action(population,
                                all_actions=all_actions,
                                epsilon=1.0)
            actions.add(act)

        # then
        assert len(actions) == all_actions

    def test_should_return_best_fitness_action(self, cfg):
        # given
        all_actions = cfg.number_of_possible_actions
        population = ClassifiersList()

        # when & then
        # C1 - does not anticipate change
        c1 = Classifier(action=1, cfg=cfg)
        population.append(c1)

        # Some random action should be selected here
        best_action = exploit(population, all_actions)
        assert best_action is not None

        # when & then
        # C2 - does anticipate some change
        c2 = Classifier(action=2,
                        effect='1###0###',
                        reward=0.25,
                        cfg=cfg)
        population.append(c2)

        # Here C2 action should be selected
        best_action = exploit(population, all_actions)
        assert best_action == 2

        # when & then
        # C3 - does anticipate change and is quite good
        c3 = Classifier(action=3,
                        effect='1#######',
                        quality=0.8,
                        reward=5,
                        cfg=cfg)
        population.append(c3)

        # Here C3 has the biggest fitness score
        best_action = exploit(population, all_actions)
        assert best_action == 3

    def test_should_return_random_action(self, cfg):
        # given
        all_actions = cfg.number_of_possible_actions
        random_actions = []

        # when
        for _ in range(0, 500):
            random_actions.append(choose_random_action(all_actions))

        min_action = min(random_actions)
        max_action = max(random_actions)

        # then
        assert min_action == 0
        assert max_action == 7

    def test_should_return_latest_action(self, cfg):
        # given
        all_actions = cfg.number_of_possible_actions
        population = ClassifiersList()
        c0 = Classifier(action=0, cfg=cfg)
        c0.talp = 1

        # when
        population.append(c0)

        # Should return first action with no classifiers
        assert 1 == choose_latest_action(population, all_actions)

        # Add rest of classifiers
        population.append(Classifier(action=3, cfg=cfg))
        population.append(Classifier(action=7, cfg=cfg))
        population.append(Classifier(action=5, cfg=cfg))
        population.append(Classifier(action=1, cfg=cfg))
        population.append(Classifier(action=4, cfg=cfg))
        population.append(Classifier(action=2, cfg=cfg))
        population.append(Classifier(action=6, cfg=cfg))

        # Assign each classifier random talp from certain range
        for cl in population:
            cl.talp = randint(70, 100)

        # But third classifier (action 7) will be the executed long time ago
        population[2].talp = randint(10, 20)

        # then
        assert choose_latest_action(population, all_actions) == 7

    def test_should_return_worst_quality_action(self, cfg):
        # given
        all_actions = cfg.number_of_possible_actions
        population = ClassifiersList()
        c0 = Classifier(action=0, cfg=cfg)
        population.append(c0)

        # Should return C1 (because it's first not mentioned)
        assert choose_action_from_knowledge_array(population, all_actions) == 1

        # Add rest of classifiers
        c1 = Classifier(action=1, numerosity=31, quality=0.72, cfg=cfg)
        population.append(c1)

        c2 = Classifier(action=2, numerosity=2, quality=0.6, cfg=cfg)
        population.append(c2)

        c3 = Classifier(action=3, numerosity=2, quality=0.63, cfg=cfg)
        population.append(c3)

        c4 = Classifier(action=4, numerosity=7, quality=0.75, cfg=cfg)
        population.append(c4)

        c5 = Classifier(action=5, numerosity=1, quality=0.63, cfg=cfg)
        population.append(c5)

        c6 = Classifier(action=6, numerosity=6, quality=0.52, cfg=cfg)
        population.append(c6)

        c7 = Classifier(action=7, numerosity=10, quality=0.36, cfg=cfg)
        population.append(c7)

        # then
        # Classifier C7 should be the worst here
        assert choose_action_from_knowledge_array(population, all_actions) == 7
