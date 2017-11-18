from gym_multiplexer.utils import get_correct_answer


class TestUtils:

    def test_should_calculate_correct_answer_for_3bit_multiplexer(self):
        assert 1 == get_correct_answer('010', 1)
        assert 0 == get_correct_answer('110', 1)

    def test_should_calculate_correct_answer_for_6bit_multiplexer(self):
        assert 0 == get_correct_answer('110100', 2)

    def test_should_calculate_correct_answer_for_11bit_multiplexer(self):
        assert 1 == get_correct_answer('101101101', 3)
