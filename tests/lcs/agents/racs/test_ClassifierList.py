import pytest

from lcs import Perception
from lcs.agents.racs import Configuration, Condition,\
    Classifier, ClassifierList
from lcs.representations import UBR


class TestClassifierList:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder_bits=4)

    def test_should_initialize_classifier_list(self, cfg):
        # given
        cl1 = Classifier(cfg=cfg)
        cl2 = Classifier(cfg=cfg)
        cl3 = Classifier(cfg=cfg)

        # when
        cll = ClassifierList(*[cl1, cl2])

        # then
        assert len(cll) == 2
        assert cl1 in cll
        assert cl2 in cll
        assert cl3 not in cll

    def test_should_form_match_set(self, cfg):
        # given
        # 4bit encoding 0.2 => 3, 0.6 => 9
        observation = Perception([0.2, 0.6], oktypes=(float,))

        cl1 = Classifier(condition=Condition([UBR(2, 5), UBR(8, 11)], cfg=cfg),
                         cfg=cfg)
        cl2 = Classifier(condition=Condition([UBR(5, 7), UBR(5, 12)], cfg=cfg),
                         cfg=cfg)
        cl3 = Classifier(cfg=cfg)

        population = ClassifierList(*[cl1, cl2, cl3])

        # when
        match_set = population.form_match_set(observation)

        # then
        assert len(match_set) == 2
        assert cl1 in match_set
        assert cl2 not in match_set
        assert cl3 in match_set

    def test_should_form_action_set(self, cfg):
        # given
        cl1 = Classifier(action=0, cfg=cfg)
        cl2 = Classifier(action=0, cfg=cfg)
        cl3 = Classifier(action=1, cfg=cfg)

        population = ClassifierList(*[cl1, cl2, cl3])

        # when
        action_set = population.form_action_set(0)

        # then
        assert len(action_set) == 2
        assert cl1 in action_set
        assert cl2 in action_set
        assert cl3 not in action_set

    def test_should_apply_reinforcement_learning(self, cfg):
        # given
        cl = Classifier(
            reward=34.29,
            intermediate_reward=11.29,
            cfg=cfg)
        population = ClassifierList(*[cl])

        # when
        population.apply_reinforcement_learning(0, 28.79)

        # then
        assert abs(33.94 - population[0].r) < 0.1
        assert abs(10.74 - population[0].ir) < 0.1
