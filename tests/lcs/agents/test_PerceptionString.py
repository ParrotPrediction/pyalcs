from lcs.agents import PerceptionString
from lcs.representations import UBR


class TestPerceptionString:

    def test_should_initialize_with_defaults(self):
        assert len(PerceptionString("foo")) == 3
        assert len(PerceptionString(['b', 'a', 'r'])) == 3

    def test_should_create_empty_with_defaults(self):
        # when
        ps = PerceptionString.empty(3)

        # then
        assert len(ps) == 3
        assert repr(ps) == '###'

    def test_should_create_empty_for_ubr(self):
        # given
        length = 3
        wildcard = UBR(0, 16)

        # when
        ps = PerceptionString.empty(length, wildcard, oktypes=(UBR,))

        # then
        assert len(ps) == 3
        assert ps[0] == ps[1] == ps[2] == wildcard
        assert ps[0] is not ps[1] is not ps[2]

    def test_should_safely_modify_single_attribute(self):
        # given
        length = 3
        wildcard = UBR(0, 16)
        ps = PerceptionString.empty(length, wildcard, oktypes=(UBR, ))

        # when
        ps[0].x1 = 2

        # then (check if objects are not stored using references)
        assert ps[1].x1 == 0
