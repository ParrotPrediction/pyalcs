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
        with pytest.raises(TypeError) as _:
            Perception(obs)
