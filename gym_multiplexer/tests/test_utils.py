from gym_multiplexer.utils import get_correct_answer


class TestUtils:

    def test_should_calculate_correct_answer_for_3bit_multiplexer(self):
        assert 1 == get_correct_answer('0100', 1)
        assert 0 == get_correct_answer('1100', 1)

    def test_should_calculate_correct_answer_for_6bit_multiplexer(self):
        assert 0 == get_correct_answer('1101000', 2)

    def test_should_calculate_correct_answer_for_11bit_multiplexer(self):
        assert 1 == get_correct_answer('1011011010', 3)
