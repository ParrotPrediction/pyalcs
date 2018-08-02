from lcs.representations import UBR


class TestUBR:

    def test_should_compare_without_ordering(self):
        # given
        o1 = UBR(0, 2)
        o2 = UBR(2, 0)

        # then
        assert o1 == o2
        assert (o1 != o2) is False
