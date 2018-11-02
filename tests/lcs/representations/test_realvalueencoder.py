from random import random

import numpy as np
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

    def test_should_encode_with_one_bit(self):
        # given
        encoder = RealValueEncoder(1)

        # then
        assert encoder.encode(0.0) == 0
        assert encoder.encode(0.5) == 0
        assert encoder.encode(0.51) == 1
        assert encoder.encode(1.0) == 1

    def test_should_encode_with_two_bits(self):
        # given
        encoder = RealValueEncoder(2)

        # then
        assert encoder.encode(0.0) == 0
        assert encoder.encode(0.33) == 1
        assert encoder.encode(0.66) == 2
        assert encoder.encode(1.0) == 3

    def test_should_encode_with_four_bits(self):
        # given
        bits = 4  # 2^bits discrete states
        encoder = RealValueEncoder(bits)

        # then
        assert encoder.encode(0.0) == 0
        assert encoder.encode(0.5) == 8
        assert encoder.encode(1.0) == 15

    def test_should_decode_values(self):
        # given
        bits = 4
        encoder = RealValueEncoder(bits)

        # then
        assert 0.0 == encoder.decode(0)
        assert abs(0.5 - encoder.decode(8)) < 0.05
        assert 1.0 == encoder.decode(15)

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

    @pytest.mark.parametrize("_bits, _min_range, _max_range", [
        (1, 0, 1),
        (2, 0, 3),
        (4, 0, 15),
        (8, 0, 255)
    ])
    def test_should_return_min_max_range(self, _bits, _min_range, _max_range):
        # given
        encoder = RealValueEncoder(_bits)

        # when
        min_val, max_val = encoder.range

        # then
        assert min_val == _min_range
        assert max_val == _max_range

    @pytest.mark.parametrize("_p", [
        0.8, np.float_(0.2), 0
    ])
    def test_encode_should_return_integer(self, _p):
        # given
        encoder = RealValueEncoder(2)

        # when
        encoded = encoder.encode(_p)

        # then
        assert type(encoded) is int
