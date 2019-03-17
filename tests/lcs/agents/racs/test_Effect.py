import pytest

from lcs import Perception
from lcs.agents.racs import Configuration, Effect
from lcs.representations import Interval


class TestEffect:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2)

    @pytest.mark.parametrize("_e, _result", [
        ([Interval(0., 1.), Interval(0., 1.)], False),
        ([Interval(0., 1.), Interval(.1, .3)], True),
        ([Interval(.4, .8), Interval(.4, .9)], True),
    ])
    def test_should_detect_change(self, _e, _result, cfg):
        assert Effect(_e, cfg).specify_change == _result

    def test_should_create_pass_through_effect(self, cfg):
        # when
        effect = Effect.pass_through(cfg)

        # then
        assert len(effect) == cfg.classifier_length
        for allele in effect:
            assert allele == cfg.classifier_wildcard

    @pytest.mark.parametrize("_p0, _p1, _effect, is_specializable", [
        # Effect is all pass-through. Can be specialized.
        ([0.5, 0.5], [0.5, 0.5], [Interval(0., 1.), Interval(0., 1.)], True),
        # 1 pass-through effect get skipped. Second effect attribute get's
        # examined. P1 perception is not in correct range. That's invalid
        ([0.5, 0.5], [0.5, 0.5], [Interval(0., 1.), Interval(.2, .4)], False),
        # In this case the range is proper, but no change is anticipated.
        # In this case this should be a pass-through symbol.
        ([0.5, 0.5], [0.5, 0.5], [Interval(0., 1.), Interval(.2, .8)], False),
        # Here second perception attribute changes. 0.8 => 12
        ([0.5, 0.5], [0.5, 0.8], [Interval(0., 1.), Interval(.75, .85)], True)
    ])
    def test_should_specialize(self, _p0, _p1, _effect, is_specializable, cfg):
        # given
        p0 = Perception(_p0, oktypes=(float,))
        p1 = Perception(_p1, oktypes=(float,))
        effect = Effect(_effect, cfg=cfg)

        # then
        assert effect.is_specializable(p0, p1) is is_specializable

    @pytest.mark.parametrize("_effect1, _effect2, _result", [
        ([Interval(0., 1.), Interval(0., 1.)],
         [Interval(.2, .4), Interval(.5, .8)], True),
        ([Interval(.7, .9), Interval(0., 1.)],
         [Interval(.2, .4), Interval(.7, .8)], False),
        ([Interval(0., 1.), Interval(.3, .7)],
         [Interval(.2, .4), Interval(.4, .8)], False),
        ([Interval(.2, .4), Interval(.5, .5)],
         [Interval(.4, .2), Interval(.5, .5)], True),
    ])
    def test_should_subsume_effect(self, _effect1, _effect2, _result, cfg):
        # given
        effect1 = Effect(_effect1, cfg=cfg)
        effect2 = Effect(_effect2, cfg=cfg)

        # then
        assert effect1.subsumes(effect2) == _result

    @pytest.mark.parametrize("_effect1, _effect2, _result", [
        ([Interval(0., 1.), Interval(0., 1.)],
         [Interval(.01, 1.), Interval(0., .99)], True),
        ([Interval(0., 1.), Interval(0., 1.)],
         [Interval(.5, .7), Interval(0., 1.)], False)
    ])
    def test_should_detect_equal(self, _effect1, _effect2, _result, cfg):
        # given
        effect1 = Effect(_effect1, cfg=cfg)
        effect2 = Effect(_effect2, cfg=cfg)

        # then
        assert (effect1 == effect2) is _result

    #
    # @pytest.mark.parametrize("_effect, _result", [
    #     ([UBR(0, 15), UBR(0, 7)], 'OOOOOOOOOO|OOOOO.....')
    # ])
    # def test_should_visualize(self, _effect, _result, cfg):
    #     assert repr(Effect(_effect, cfg=cfg)) == _result
