from lcs.agents.acs2 import ProbabilityEnhancedAttribute


class TestProbabilityEnhancedAttribute:

    def test_should_initialize_correctly_str(self):
        attr = ProbabilityEnhancedAttribute('1')
        assert attr.sum_of_probabilities() == 1.0
        assert attr['1'] == 1.0

    def test_should_initialize_correctly_dict(self):
        attr = ProbabilityEnhancedAttribute({'1': 1.0, '0': 0.0})
        assert attr.sum_of_probabilities() == 1.0
        assert attr['1'] == 1.0

    def test_is_similar_0(self):
        # given
        attr = ProbabilityEnhancedAttribute({'1': 0.8, '0': 0.2})

        # then
        assert attr.is_similar(attr)

    def test_is_similar_1(self):
        # given
        attr1 = ProbabilityEnhancedAttribute({'1': 0.8, '0': 0.2})
        attr2 = ProbabilityEnhancedAttribute({'1': 0.4, '0': 0.6})

        # then
        assert attr1.is_similar(attr2)
        assert attr2.is_similar(attr1)

    def test_is_similar_2(self):
        # given
        attr1 = ProbabilityEnhancedAttribute({'0': 0.5, '1': 0.4, '9': 0.1})
        attr2 = ProbabilityEnhancedAttribute({'0': 0.1, '1': 0.8, '9': 0.1})

        # then
        assert attr1.is_similar(attr2)
        assert attr2.is_similar(attr1)

    def test_is_similar_3(self):
        # given
        attr1 = ProbabilityEnhancedAttribute({'0': 0.5, '1': 0.4, '9': 0.1})
        attr2 = ProbabilityEnhancedAttribute({'0': 0.9, '9': 0.1})

        # then
        assert not attr1.is_similar(attr2)
        assert not attr2.is_similar(attr1)

    def test_is_compact(self):
        # given
        attr1 = ProbabilityEnhancedAttribute({'0': 0.6, '1': 0.4, '9': 0.0})
        attr2 = ProbabilityEnhancedAttribute({'0': 0.6, '1': 0.4})

        # then
        assert not attr1.is_compact()
        assert attr2.is_compact()

    def test_make_compact(self):
        # given
        attr1 = ProbabilityEnhancedAttribute({'0': 0.6, '1': 0.4, '9': 0.0})
        attr2 = ProbabilityEnhancedAttribute({'0': 0.6, '1': 0.4})

        # when
        attr1.make_compact()

        # then
        assert attr1.is_compact()
        assert attr1 == attr2

    def test_get_best_symbol(self):
        # given
        attr1 = ProbabilityEnhancedAttribute({'0': 0.4, '1': 0.6})
        attr2 = ProbabilityEnhancedAttribute({'0': 0.5, '1': 0.3, '9': 0.2})
        attr3 = ProbabilityEnhancedAttribute({'0': 0.3, '1': 0.3, '9': 0.4})

        # then
        assert attr1.get_best_symbol() == '1'
        assert attr2.get_best_symbol() == '0'
        assert attr3.get_best_symbol() == '9'

    def test_insert_symbol(self):
        # given
        attr1 = ProbabilityEnhancedAttribute({'0': 0.5, '1': 0.5})
        attr2 = ProbabilityEnhancedAttribute({'0': 0.4, '1': 0.4, '9': 0.2})

        # when
        attr1.insert_symbol('9', 0.8, 0.2)

        # then
        assert attr1 == attr2

    def test_remove_symbol(self):
        # given
        attr1 = ProbabilityEnhancedAttribute({'0': 0.4, '1': 0.4, '9': 0.2})
        attr2 = ProbabilityEnhancedAttribute({'0': 0.5, '1': 0.5})

        # when
        assert attr1.remove_symbol('9')

        # then
        assert attr1 == attr2

    def test_remove_last_symbol(self):
        # given
        attr1 = ProbabilityEnhancedAttribute({'0': 1.0})
        attr2 = ProbabilityEnhancedAttribute({'0': 1.0})

        # when
        assert not attr1.remove_symbol('0')

        # then
        assert attr1 == attr2
