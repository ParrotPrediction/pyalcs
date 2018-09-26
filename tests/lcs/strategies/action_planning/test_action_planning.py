from random import randint

import pytest

from lcs import Perception
from lcs.agents.acs2 import Configuration, ClassifiersList, Classifier
from lcs.strategies.action_planning.action_planning import \
    get_quality_classifiers_list, exists_classifier


class TestActionPlanning:

    @pytest.fixture
    def cfg(self):
        return Configuration(8, 8)

    def test_get_quality_classifiers_list(self, cfg):
        # given
        population = ClassifiersList()

        # C1 - matching
        c1 = Classifier(quality=0.9, cfg=cfg)

        # C2 - matching
        c2 = Classifier(quality=0.7, cfg=cfg)

        # C3 - non-matching
        c3 = Classifier(quality=0.5, cfg=cfg)

        # C4 - non-matching
        c4 = Classifier(quality=0.1, cfg=cfg)

        population.append(c1)
        population.append(c2)
        population.append(c3)
        population.append(c4)

        # when
        match_set = get_quality_classifiers_list(population, 0.5, cfg)

        # then
        assert 2 == len(match_set)
        assert c1 in match_set
        assert c2 in match_set

    def test_exists_classifier(self, cfg):
        # given
        population = ClassifiersList()
        prev_situation = Perception('01100000')
        situation = Perception('11110000')
        act = 0
        q = 0.5

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
        result0 = exists_classifier(population,
                                    previous_situation=prev_situation,
                                    situation=situation, action=act, quality=q)

        population.append(c1)
        result1 = exists_classifier(population,
                                    previous_situation=prev_situation,
                                    situation=situation, action=act, quality=q)

        # then
        assert result0 is False
        assert result1 is True
