import pytest

from lcs import Perception
from lcs.agents.acs2 import Configuration, ClassifiersList, Classifier
from lcs.strategies.action_planning.action_planning import \
    suitable_cl_exists, search_goal_sequence


class TestActionPlanning:

    @pytest.fixture
    def cfg(self):
        return Configuration(8, 8, theta_r=0.9)

    def test_should_find_suitable_classifier(self, cfg):
        # given
        cfg.theta_r = 0.5
        population = ClassifiersList()
        prev_situation = Perception('01100000')
        situation = Perception('11110000')
        act = 0

        # C1 - OK
        c1 = Classifier(condition='0##0####', action=0, effect='1##1####',
                        quality=0.7, cfg=cfg)

        # C2 - wrong action
        c2 = Classifier(condition='0##0####', action=1, effect='1##1####',
                        quality=0.7, cfg=cfg)

        # C3 - wrong condition
        c3 = Classifier(condition='0##1####', action=0, effect='1##1####',
                        quality=0.7, cfg=cfg)

        # C4 - wrong effect
        c4 = Classifier(condition='0##0####', action=0, effect='1##0####',
                        quality=0.7, cfg=cfg)

        # C5 - wrong quality
        c5 = Classifier(condition='0##0####', action=0, effect='1##1####',
                        quality=0.25, cfg=cfg)

        population.append(c2)
        population.append(c3)
        population.append(c4)
        population.append(c5)

        # when
        result0 = suitable_cl_exists(population,
                                     p0=prev_situation,
                                     p1=situation, action=act)

        population.append(c1)
        result1 = suitable_cl_exists(population,
                                     p0=prev_situation,
                                     p1=situation, action=act)

        # then
        assert result0 is False
        assert result1 is True

    def test_search_goal_sequence_1(self, cfg):
        # given
        start = Perception('01111111')
        goal = Perception('00111111')

        classifiers = ClassifiersList(
            Classifier(condition="#1######", action=1, effect="#0######",
                       quality=0.88, cfg=cfg),
            Classifier(condition="#1######", action=1, effect="#0######",
                       quality=0.92, cfg=cfg)
        )

        # when
        result = search_goal_sequence(classifiers, start, goal)

        # then
        assert result == [1]

    def test_search_goal_sequence_2(self, cfg):
        # given
        start = Perception('01111111')
        goal = Perception('00111111')

        classifiers = ClassifiersList(
            Classifier(condition="#1######", action=1, effect="#0######",
                       quality=0.88, cfg=cfg),
            Classifier(condition="#0######", action=1, effect="#1######",
                       quality=0.98, cfg=cfg)
        )

        # when
        result = search_goal_sequence(classifiers, start, goal)

        # then
        assert result == []

    def test_search_goal_sequence_3(self, cfg):
        # given
        start = Perception('01111111')
        goal = Perception('10111111')

        classifiers = ClassifiersList(
            Classifier(condition="#1######", action=1, effect="#0######",
                       quality=0.94, cfg=cfg),
            Classifier(condition="0#######", action=2, effect="1#######",
                       quality=0.98, cfg=cfg),
        )

        # when
        result = search_goal_sequence(classifiers, start, goal)

        # then
        assert len(result) == 2
        assert 1 in result
        assert 2 in result
