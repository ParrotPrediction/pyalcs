import random

import numpy as np
import pytest

from lcs.representations.RealValueEncoder import RealValueEncoder


class TestRealValueEncoder:

    @pytest.mark.parametrize("_bits, _val, _encoded", [
        (2, -0.5, 0),
        (2, 1.0, 3),
        (2, 2.0, 3),
        (2, 3.0, 3),
    ])
    def test_should_clip_values_outside_range(self, _bits, _val, _encoded):
        assert RealValueEncoder(_bits).encode(_val) == _encoded

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

    @pytest.mark.parametrize("_bits, _val, _encoded", [
        (1, 0.0, 0), (1, 0.5, 0), (1, 0.51, 1), (1, 1.0, 1),
        (2, 0.0, 0), (2, 0.33, 1), (2, 0.66, 2), (2, 1.0, 3),
        (4, 0.0, 0), (4, 0.5, 8), (4, 1.0, 15),
    ])
    def test_should_encode(self, _bits, _val, _encoded):
        assert RealValueEncoder(_bits).encode(_val) == _encoded

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
        observation = random.random()

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

    def test_should_encode_with_noise_added(self):
        # given
        encoder = RealValueEncoder(16)
        noise_max = 0.1

        for _ in range(100):
            # when
            val = random.random()
            encoded = encoder.encode(val)
            encoded_with_noise = encoder.encode(val, noise_max)

            # then
            assert encoded_with_noise <= encoder.range[1]
            assert encoded_with_noise >= encoded

    def test_should_encode_with_noise_subtracted(self):
        # given
        encoder = RealValueEncoder(16)
        noise_max = -0.1

        for _ in range(100):
            # when
            val = random.random()
            encoded = encoder.encode(val)
            encoded_with_noise = encoder.encode(val, noise_max)

            # then
            assert encoded_with_noise >= encoder.range[0]
            assert encoded_with_noise <= encoded
