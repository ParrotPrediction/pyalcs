import pytest

from lcs.agents.xncs import Backpropagation, Configuration, Classifier, Effect, ClassifiersList
from lcs.agents.xcs import Condition


class TestBackpropagation:


    @pytest.fixture
    def cfg(self):
        return Configuration(lmc=2, lem=0.2, number_of_actions=4)

    def test_init(self, cfg):
        bp = Backpropagation(cfg)
        assert id(bp.cfg) == id(cfg)

    def test_insert(self, cfg):
        bp = Backpropagation(cfg)
        cl = Classifier(cfg=cfg, condition=Condition("1111"), action=0, time_stamp=0)
        ef = Effect("0110")
        bp.insert_into_bp(cl, ef)
        assert id(bp.classifiers_for_update[0]) == id(cl)
        assert id(bp.update_vectors[0]) == id(ef)
        assert bp.classifiers_for_update[0] == cl
        assert bp.update_vectors[0] == ef

    def test_update(self, cfg):
        bp = Backpropagation(cfg)
        cl = Classifier(cfg=cfg, condition=Condition("1111"), action=0, time_stamp=0)
        ef = Effect("0110")
        bp.insert_into_bp(cl, ef)
        bp.update_bp()
        assert cl.effect == ef
        bp.insert_into_bp(cl, ef)
        bp.update_bp()
        assert cl.effect == ef
        assert cl.error != cfg.initial_error

    def test_update(self, cfg):
        bp = Backpropagation(cfg)
        cl = Classifier(cfg=cfg, condition=Condition("1111"), action=0, time_stamp=0)
        ef = Effect("0110")
        bp.insert_into_bp(cl, ef)
        bp.check_and_update()
        assert cl.effect is None
        bp.check_and_update()
        assert cl.effect is not None
