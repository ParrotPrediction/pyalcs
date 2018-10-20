import pytest

from lcs.agents.racs import RACS, Configuration, ClassifierList, \
    Classifier, Condition
from lcs.representations import UBR
from lcs.representations.RealValueEncoder import RealValueEncoder


class TestRACS:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder=RealValueEncoder(4))

    def test_regions_averaging(self, cfg):
        # given
        cl1 = Classifier(condition=Condition([UBR(2, 3), UBR(4, 5)], cfg),
                         cfg=cfg)
        cl2 = Classifier(condition=Condition([UBR(0, 3), UBR(4, 9)], cfg),
                         cfg=cfg)
        cl3 = Classifier(condition=Condition([UBR(1, 3), UBR(4, 15)], cfg),
                         cfg=cfg)
        cl4 = Classifier(condition=Condition([UBR(0, 13), UBR(0, 15)], cfg),
                         cfg=cfg)
        population = ClassifierList(*[cl1, cl2, cl3, cl4])
        agent = RACS(cfg, population)

        # when
        result = agent._count_averaged_regions()

        # then
        assert type(result) is dict
        assert result == {1: 0.5, 2: 0.25, 3: 0.125, 4: 0.125}
