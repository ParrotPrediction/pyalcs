from bitstring import BitString, BitArray


def get_correct_answer(bitstring: str, control_bits: int) -> int:
    bits = BitString(bin=bitstring)

    _ctrl_bits = bits[:control_bits]
    _data_bits = bits[control_bits:]

    return int(_data_bits[_ctrl_bits.uint])
