import pytest

from lcs.agents.racs import Configuration, Classifier, Condition, Effect, \
    ClassifierList
from lcs.agents.racs.components.merging import merge, _can_be_extended, _extend
from lcs.representations import UBR
from lcs.representations.RealValueEncoder import RealValueEncoder


class TestMerging:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder=RealValueEncoder(4))

    def test_should_merge_classifier(self, cfg):
        # given
        cl = Classifier(
            condition=Condition([UBR(2, 4), UBR(6, 9)], cfg),
            action=1,
            effect=Effect([UBR(2, 4), UBR(6, 9)], cfg),
            cfg=cfg)

        e_cl1 = Classifier(
            condition=Condition([UBR(0, 13), UBR(2, 7)], cfg),
            action=1,
            effect=Effect([UBR(0, 13), UBR(2, 7)], cfg),
            cfg=cfg)

        e_cl2 = Classifier(
            condition=Condition([UBR(0, 2), UBR(6, 9)], cfg),
            action=1,
            effect=Effect([UBR(0, 2), UBR(6, 9)], cfg),
            cfg=cfg)

        pop = ClassifierList(*[e_cl1, e_cl2])

        # when
        children = list(merge(cl, pop))

        # then
        # cl + e_cl2 should be merged
        assert len(children) == 1
        assert children[0].condition == Condition([UBR(0, 4), UBR(6, 9)], cfg)
        assert children[0].effect == Condition([UBR(0, 4), UBR(6, 9)], cfg)

    @pytest.mark.parametrize("new_cl_ps, existing_cl_ps, result", [
        # the same classifiers, other mechanism should detect that
        ([UBR(1, 2), UBR(4, 5)], [UBR(1, 2), UBR(4, 5)], True),
        # new_cl extends existing_cl (one ubr)
        ([UBR(1, 3), UBR(4, 5)], [UBR(1, 2), UBR(4, 5)], True),
        # all invalid
        ([UBR(4, 6), UBR(1, 2)], [UBR(1, 2), UBR(4, 5)], False),
        # one invalid, one mergable
        ([UBR(1, 4), UBR(7, 8)], [UBR(1, 2), UBR(4, 5)], False),
        # subsumption is not merging
        ([UBR(1, 4), UBR(4, 5)], [UBR(2, 3), UBR(4, 5)], False),
    ])
    def test_if_classifier_can_be_extended(
            self, new_cl_ps, existing_cl_ps, result, cfg):
        # given
        new_cl = Classifier(
            condition=Condition(new_cl_ps, cfg),
            effect=Effect(new_cl_ps, cfg),
            cfg=cfg)

        existing_cl = Classifier(
            condition=Condition(existing_cl_ps, cfg),
            effect=Effect(existing_cl_ps, cfg),
            cfg=cfg)

        # then
        assert _can_be_extended(new_cl, existing_cl) == result

    @pytest.mark.parametrize("ps1, ps2, result", [
        ([UBR(2, 3), UBR(2, 2)],
         [UBR(2, 5), UBR(2, 2)],
         [UBR(2, 5), UBR(2, 2)]),
        ([UBR(5, 6), UBR(2, 3)],
         [UBR(3, 6), UBR(2, 4)],
         [UBR(3, 6), UBR(2, 4)]),
    ])
    def test_should_extend_with_other_classifier(self, ps1, ps2, result, cfg):
        # given
        new_cl = Classifier(
            condition=Condition(ps1, cfg),
            effect=Effect(ps1, cfg),
            cfg=cfg)
        existing_cl = Classifier(
            condition=Condition(ps2, cfg),
            effect=Effect(ps2, cfg),
            cfg=cfg)

        assert _can_be_extended(new_cl, existing_cl) is True

        # when
        cl = _extend(new_cl, existing_cl)

        # then
        assert cl is not None
        assert cl.condition == Condition(result, cfg)
        assert cl.effect == Effect(result, cfg)
