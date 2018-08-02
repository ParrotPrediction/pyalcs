from lcs import Perception
from lcs.agents.racs import Classifier, ClassifierList


class TestClassifierList:
    def test_should_initialize_classifier_list(self):
        # given
        cl1 = Classifier()
        cl2 = Classifier()
        cl3 = Classifier()

        # when
        cll = ClassifierList(*[cl1, cl2])

        # then
        assert len(cll) == 2
        assert cl1 in cll
        assert cl2 in cll
        assert cl3 not in cll

    def test_should_form_match_set(self):
        # given
        p = Perception([0.2, 0.6], oktypes=(float,))

        # handle wildcards - first assumption
