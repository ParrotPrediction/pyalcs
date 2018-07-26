from random import random

import pytest

from lcs.representations.RealValueEncoder import RealValueEncoder


class TestRealValueEncoder:

    def test_should_deny_illegal_values_when_encoding(self):
        # given
        encoder = RealValueEncoder(2)

        # when
        with pytest.raises(ValueError) as e1:
            encoder.encode(-0.1)

        with pytest.raises(ValueError) as e2:
            encoder.encode(1.1)

        # then
        assert e1 is not None
        assert e2 is not None

    def test_should_deny_illegal_values_when_decoding(self):
        # given
        encoder = RealValueEncoder(2)

        # when
        with pytest.raises(ValueError) as e1:
            encoder.decode(-1)

        with pytest.raises(ValueError) as e2:
            encoder.decode(5)

        # then
        assert e1 is not None
        assert e2 is not None

    def test_should_encode_values(self):
        # given
        bits = 4  # 2^bits discrete states
        encoder = RealValueEncoder(bits)

        # then
        assert 0 == encoder.encode(0.0)
        assert 8 == encoder.encode(0.5)
        assert 16 == encoder.encode(1.0)

    def test_should_decode_values(self):
        # given
        bits = 4
        encoder = RealValueEncoder(bits)

        # then
        assert 0.0 == encoder.decode(0)
        assert 0.5 == encoder.decode(8)
        assert 1.0 == encoder.decode(16)

    def test_should_encode_and_decode_approximately(self):
        # given
        encoder = RealValueEncoder(8)
        epsilon = 0.01
        observation = random()

        # when
        encoded = encoder.encode(observation)
        decoded = encoder.decode(encoded)

        # then
        assert abs(observation - decoded) < epsilon
