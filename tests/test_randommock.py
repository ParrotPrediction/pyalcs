from tests.randommock import RandomMock, SampleMock


class TestRandomTest:

    def test_randommock_returns_values_in_a_given_sequence(self):
        f = RandomMock([0.1, 0.2, 0.3])
        assert 0.1 == f()
        assert 0.2 == f()
        assert 0.3 == f()

    def test_samplemock_returns_list_elements_in_a_given_sequence_1(self):
        sample_func = SampleMock([2, 0, 1])
        assert [15, 3, 14] == sample_func([3, 14, 15], 3)

    def test_samplemock_returns_list_elements_in_a_given_sequence_2(self):
        sample_func = SampleMock([2, 0, 1])
        assert [15, 3, 14] == sample_func([3, 14, 15, 92, 6], 3)

    def test_samplemock_returns_list_elements_in_a_given_sequence_3(self):
        sample_func = SampleMock([1, 15, 2, 15])
        assert [14, 3, 15] == sample_func([3, 14, 15], 3)

    def test_testsample4(self):
        sample_func = SampleMock([10, 2, 1, 15])
        assert [3, 15, 14] == sample_func([3, 14, 15, 92, 6], 3)
