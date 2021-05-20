import pytest

from lcs.agents.xncs import Backpropagation, Configuration, Classifier, Effect, ClassifiersList
from lcs.agents.xcs import Condition


class TestBackpropagation:


    @pytest.fixture
    def cfg(self):
        return Configuration(lmc=2, lem=0.2, number_of_actions=4)

    @pytest.fixture
    def situation(self):
        return "1100"

    @pytest.fixture
    def classifiers_list_diff_actions(self, cfg, situation):
        classifiers_list = ClassifiersList(cfg)
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 0, 0, Effect(situation)))
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 1, 0, Effect(situation)))
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 2, 0, Effect(situation)))
        classifiers_list.insert_in_population(Classifier(cfg, Condition(situation), 3, 0, Effect(situation)))
        return classifiers_list

    @pytest.fixture
    def bp(self, cfg, classifiers_list_diff_actions):
        return Backpropagation(cfg, classifiers_list_diff_actions)

    def test_init(self, cfg, bp):
        assert id(bp.cfg) == id(cfg)

    def test_insert(self, cfg, bp):
        cl = Classifier(cfg=cfg, condition=Condition("1111"), action=0, time_stamp=0)
        ef = Effect("0110")
        bp.insert_into_bp(cl, ef)
        assert id(bp.classifiers_for_update[0]) == id(cl)
        assert id(bp.update_vectors[0]) == id(ef)
        assert bp.classifiers_for_update[0] == cl
        assert bp.update_vectors[0] == ef

    def test_update(self, cfg, bp):
        cl = Classifier(cfg=cfg, condition=Condition("1111"), action=0, time_stamp=0)
        ef = Effect("0110")
        bp.insert_into_bp(cl, ef)
        bp.update_bp()
        assert cl.effect == ef
        bp.insert_into_bp(cl, ef)
        bp.update_bp()
        assert cl.effect == ef
        assert cl.error != cfg.initial_error

