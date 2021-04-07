import pytest

from lcs import Perception


class TestPerception:

    def test_should_handle_str_state(self):
        # given
        obs = "foo"

        # when
        p = Perception(obs)

        # then
        assert len(p) == 3
        assert p[0] == 'f'
        assert p[1] == 'o'
        assert p[2] == 'o'

    def test_should_handle_list_state(self):
        # given
        obs = ['f', 'o', 'o']

        # when
        p = Perception(obs)

        # then
        assert len(p) == 3
        assert p[0] == 'f'
        assert p[1] == 'o'
        assert p[2] == 'o'

    def test_should_list_float_state(self):
        # given
        obs = [1.1, 1.2, 1.3]

        # when
        p = Perception(obs, oktypes=(float,))

        # then
        assert len(p) == 3
        assert p[0] == 1.1
        assert p[1] == 1.2
        assert p[2] == 1.3

    def test_should_fail_on_invalid_type(self):
        # given
        obs = ["f", "o", 0]

        # when & then
        with pytest.raises(AssertionError) as _:
            Perception(obs)

    def test_should_create_empty_perception(self):
        # when
        p = Perception.empty()

        # then
        assert p is not None

    def test_should_compare_equal_hashes(self):
        assert Perception("111") == Perception("111")
        assert Perception("111") is not Perception("111")
        assert hash(Perception("111")) == hash(Perception("111"))

    @pytest.mark.parametrize('_p1, _p2, _res', [
        ('1111', '1111', True),
        ('1111', '1112', False),
        ('1111', '2222', False),
    ])
    def test_should_detect_equal_perceptions(self, _p1, _p2, _res):
        if _res:
            assert _p1 == _p2
        else:
            assert _p1 != _p2
