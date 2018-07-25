import pytest

from random import randint

from lcs.acs2 import ACS2Configuration, Classifier, ClassifiersList
from lcs.strategies.action_selection import exploit, choose_random_action, \
    choose_latest_action, choose_action_from_knowledge_array


class TestActionSelection:

    @pytest.fixture
    def cfg(self):
        return ACS2Configuration(8, 8)

    def test_should_return_best_fitness_action(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)

        # when & then
        # C1 - does not anticipate change
        c1 = Classifier(action=1, cfg=cfg)
        population.append(c1)

        # Some random action should be selected here
        best_action = exploit(population)
        assert best_action is not None

        # when & then
        # C2 - does anticipate some change
        c2 = Classifier(action=2,
                        effect='1###0###',
                        reward=0.25,
                        cfg=cfg)
        population.append(c2)

        # Here C2 action should be selected
        best_action = exploit(population)
        assert 2 == best_action

        # when & then
        # C3 - does anticipate change and is quite good
        c3 = Classifier(action=3,
                        effect='1#######',
                        quality=0.8,
                        reward=5,
                        cfg=cfg)
        population.append(c3)

        # Here C3 has the biggest fitness score
        best_action = exploit(population)
        assert 3 == best_action

    def test_should_return_random_action(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)
        random_actions = []

        # when
        for _ in range(0, 500):
            random_actions.append(choose_random_action(population))

        min_action = min(random_actions)
        max_action = max(random_actions)

        # then
        assert 0 == min_action
        assert 7 == max_action

    def test_should_return_latest_action(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)
        c0 = Classifier(action=0, cfg=cfg)
        c0.talp = 1

        # when
        population.append(c0)

        # Should return first action with no classifiers
        assert 1 == choose_latest_action(population)

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
        assert 7 == choose_latest_action(population)

    def test_should_return_worst_quality_action(self, cfg):
        # given
        population = ClassifiersList(cfg=cfg)
        c0 = Classifier(action=0, cfg=cfg)
        population.append(c0)

        # Should return C1 (because it's first not mentioned)
        assert 1 == choose_action_from_knowledge_array(population)

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
        assert 7 == choose_action_from_knowledge_array(population)
